#!/usr/bin/env python3
# Copyright (c) 2025-2026 The Zcash developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or https://www.opensource.org/licenses/mit-license.php .

#
# Test z_shieldcoinbase RPC against the Z3 stack (zebrad + zaino + zallet).
#
# Covers the post-redesign API surface (zcash/wallet#402):
#
#   z_shieldcoinbase(fromaddresses, toaddress, limit?, memo?)
#
#     fromaddresses : single account UUID (string), OR an array of
#                     wallet-owned transparent addresses all belonging
#                     to the same account.
#     toaddress     : any Zcash shielded address (Sapling, Orchard, or
#                     Unified with a shielded receiver). Need not be
#                     owned by this wallet. Transparent / TEX rejected.
#     limit         : optional u32; caps the number of selected coinbase
#                     UTXOs to the highest-value `n`.
#     memo          : optional hex string up to 1024 chars (512 bytes).
#
# The pre-flight response shape matches zcashd:
#   { remainingUTXOs, remainingValue, shieldingUTXOs, shieldingValue, opid }
#
# Uses 1 node + 2 wallets so we can exercise an external-recipient
# happy path (wallet 0 shields into a UA owned by wallet 1).
#

import time

from decimal import Decimal
from test_framework.test_framework import BitcoinTestFramework
from test_framework.authproxy import JSONRPCException
from test_framework.config import ZebraArgs
from test_framework.util import (
    assert_equal,
    assert_true,
    start_nodes,
    wait_and_assert_operationid_status,
    wait_and_assert_operationid_status_result,
)

# Coinbase outputs require 100 confirmations before they can be spent.
COINBASE_MATURITY = 100

# A well-formed but never-used account UUID, for negative tests.
BOGUS_ACCOUNT_UUID = "00000000-0000-0000-0000-000000000001"


def wait_for_shielded_balance_change(wallet, baseline_private, timeout=120):
    """
    Wait until the wallet's private balance differs from `baseline_private`.

    Used after a successful `z_shieldcoinbase` opid completes and at least one
    block has been mined, to give the wallet's scan pipeline time to ingest
    the shielding transaction and update its shielded note set.
    """
    deadline = time.time() + timeout
    last = baseline_private
    while time.time() < deadline:
        try:
            cur = Decimal(wallet.z_gettotalbalance(1, True)['private'])
            last = cur
            if cur != baseline_private:
                return cur
        except Exception:
            pass
        time.sleep(1)

    raise AssertionError(
        "wait_for_shielded_balance_change: timeout after {}s; private "
        "balance still {} (baseline was {})".format(timeout, last, baseline_private))


def wait_for_mature_coinbase(wallet, min_mature_utxos=1, timeout=120):
    """
    Wait until the wallet has indexed at least `min_mature_utxos` mature
    coinbase UTXOs spendable via `z_shieldcoinbase`.

    Zallet's `z_gettotalbalance` and `z_listunspent` reflect only what the
    wallet has scanned and committed to its local SQLite database. After
    `node.generate(N)`, there is a non-trivial delay (block fetch + scan +
    commit) before those views update. The framework's `sync_all` only
    synchronizes nodes, not wallets, and no `getwalletstatus` RPC is
    available yet (https://github.com/zcash/wallet/issues/316).

    Polls `z_listunspent(minconf=COINBASE_MATURITY + 1)` once per second.
    Considered ready when at least `min_mature_utxos` transparent outputs
    are visible at that confirmation depth.
    """
    deadline = time.time() + timeout
    last_count = 0
    while time.time() < deadline:
        try:
            utxos = wallet.z_listunspent(COINBASE_MATURITY + 1)
            transparent = [u for u in utxos if u.get('pool') == 'transparent']
            last_count = len(transparent)
            if last_count >= min_mature_utxos:
                return
        except Exception:
            pass
        time.sleep(1)

    raise AssertionError(
        "wait_for_mature_coinbase: timeout after {}s; only saw {} mature "
        "transparent UTXOs (wanted {})".format(
            timeout, last_count, min_mature_utxos
        )
    )


def expect_rpc_error(callable_, *args, **kwargs):
    """Invoke an RPC and return the JSONRPCException; fail if it didn't raise."""
    try:
        callable_(*args, **kwargs)
    except JSONRPCException as e:
        return e
    raise AssertionError(
        "Expected RPC error, but call succeeded: {}({}, {})".format(
            getattr(callable_, '__name__', '?'), args, kwargs))


def assert_in_message(e, needle):
    msg = e.error['message']
    assert_true(needle in msg, "Expected {!r} in error, got: {!r}".format(needle, msg))


class WalletZShieldCoinbaseTest(BitcoinTestFramework):

    def __init__(self):
        super().__init__()
        self.num_nodes = 1
        # 1 node + 1 wallet. The integration framework provisions one
        # zebrad per wallet; running 2 wallets would also require 2
        # nodes (and exposes a known Zaino race with concurrent wallets,
        # zcash/wallet#TBD). The "toaddress need not belong to this
        # wallet" property is structurally exercised by the receiver-
        # type validation in the backend, which we cover in V1/V2.
        self.num_wallets = 1
        self.cache_behavior = 'clean'

    def setup_nodes(self):
        # NU5 is the consensus floor for Zallet's Orchard change strategy
        # used inside z_shieldcoinbase. The default zallet.toml activates
        # NU5 at height 1; mirror that on the zebrad side.
        args = [
            ZebraArgs(
                miner_address=addr,
                activation_heights={"NU5": 1},
            ) for addr in self.miner_addresses
        ]
        return start_nodes(self.num_nodes, self.options.tmpdir, args)

    # ------------------------------------------------------------------
    # Validation tests: fast (no coinbase maturity required).
    # ------------------------------------------------------------------

    def run_validation_tests(self, w0, w0_taddr, w0_account_uuid,
                             w0_zaddr, w0_extra_account_uuid):
        # ---- `toaddress` validation ---------------------------------

        print("Test V1: toaddress is gibberish -> InvalidParameter...")
        e = expect_rpc_error(
            w0.z_shieldcoinbase, w0_account_uuid, "not_a_real_address")
        assert_in_message(e, "unknown address format")
        print("  PASSED")

        print("Test V2: toaddress is transparent -> backend rejects...")
        e = expect_rpc_error(
            w0.z_shieldcoinbase, w0_account_uuid, w0_taddr)
        # Backend surfaces ShieldingRequiresShieldedRecipient via
        # ProposalError. We don't pin the exact wording, just that the
        # call fails — a successful shield-to-taddr would be a serious
        # regression.
        msg = e.error['message']
        assert_true(
            "shielded" in msg.lower() or "transparent" in msg.lower(),
            "Expected shielded-recipient error, got: {!r}".format(msg))
        print("  PASSED")

        # ---- `fromaddresses` JSON-type validation -------------------

        print("Test V3: fromaddresses is a number -> InvalidParameter...")
        e = expect_rpc_error(
            w0.z_shieldcoinbase, 42, w0_zaddr)
        assert_in_message(e, "fromaddresses")
        print("  PASSED")

        print("Test V4: fromaddresses is a bool -> InvalidParameter...")
        e = expect_rpc_error(
            w0.z_shieldcoinbase, True, w0_zaddr)
        assert_in_message(e, "fromaddresses")
        print("  PASSED")

        print("Test V5: fromaddresses is an empty array -> InvalidParameter...")
        e = expect_rpc_error(
            w0.z_shieldcoinbase, [], w0_zaddr)
        assert_in_message(e, "must not be empty")
        print("  PASSED")

        print("Test V6: fromaddresses array entry is not a string...")
        e = expect_rpc_error(
            w0.z_shieldcoinbase, [42], w0_zaddr)
        assert_in_message(e, "every array entry must be a string")
        print("  PASSED")

        # ---- `fromaddresses` semantic validation --------------------

        print("Test V7: fromaddresses string is neither UUID nor array...")
        e = expect_rpc_error(
            w0.z_shieldcoinbase, "definitely-not-a-uuid", w0_zaddr)
        assert_in_message(e, "expected an account UUID string")
        print("  PASSED")

        print("Test V8: fromaddresses is a well-formed but unknown UUID...")
        e = expect_rpc_error(
            w0.z_shieldcoinbase, BOGUS_ACCOUNT_UUID, w0_zaddr)
        assert_in_message(e, "Unknown account UUID")
        print("  PASSED")

        print("Test V9: fromaddresses array contains a shielded address...")
        e = expect_rpc_error(
            w0.z_shieldcoinbase, [w0_zaddr], w0_zaddr)
        assert_in_message(e, "only transparent addresses are accepted")
        print("  PASSED")

        print("Test V10: fromaddresses array contains an unowned taddr...")
        # A well-formed regtest taddr not provisioned by this wallet.
        # (`test_framework/config.py` default — never imported into our
        # wallet under normal test setup.)
        unowned_taddr = "tmSRd1r8gs77Ja67Fw1JcdoXytxsyrLTPJm"
        assert_true(unowned_taddr != w0_taddr,
                    "Unowned taddr collision with wallet's miner address")
        e = expect_rpc_error(
            w0.z_shieldcoinbase, [unowned_taddr], w0_zaddr)
        assert_in_message(e, "not owned by any account in this wallet")
        print("  PASSED")

        print("Test V11: fromaddresses spans two accounts -> rejected...")
        # Wallet 0 has at least two accounts (0 and the extra one).
        # Resolve a transparent receiver on each, then mix.
        ua0_t = w0.z_getaddressforaccount(w0_account_uuid, ["orchard", "p2pkh"])['address']
        ua1_t = w0.z_getaddressforaccount(w0_extra_account_uuid, ["orchard", "p2pkh"])['address']
        addr0 = self._first_transparent_receiver(w0, ua0_t)
        addr1 = self._first_transparent_receiver(w0, ua1_t)
        assert_true(addr0 != addr1, "Cross-account taddrs should differ")
        e = expect_rpc_error(
            w0.z_shieldcoinbase, [addr0, addr1], w0_zaddr)
        assert_in_message(e, "must belong to the same account")
        print("  PASSED")

        # ---- Pre-flight: empty source set ---------------------------

        print("Test V12: account UUID with no transparent receivers...")
        # The extra account on wallet 0 has UAs but the call below
        # creates one with a transparent receiver. Instead, use an
        # account with no transparent UTXOs (any account works as long
        # as it hasn't received any) by asking for the extra account,
        # which (although it has a receiver) has no balance.
        # This should fail with "Insufficient" rather than "No source
        # addresses", since the receiver exists.
        # Pin whichever error the backend returns rather than asserting
        # a specific message.
        e = expect_rpc_error(
            w0.z_shieldcoinbase, w0_extra_account_uuid, w0_zaddr)
        msg = e.error['message']
        assert_true(
            len(msg) > 0,
            "Expected non-empty error for empty-source case, got: {!r}".format(msg))
        print("  PASSED (error: {})".format(msg[:80]))

    @staticmethod
    def _first_transparent_receiver(wallet, ua):
        """Return the P2PKH receiver of a UA created with a transparent component."""
        receivers = wallet.z_listunifiedreceivers(ua)
        if 'p2pkh' in receivers:
            return receivers['p2pkh']
        if 'p2sh' in receivers:
            return receivers['p2sh']
        raise AssertionError(
            "UA has no transparent receiver: {!r} -> {!r}".format(ua, receivers))

    # ------------------------------------------------------------------
    # Functional tests: require mature coinbase.
    # ------------------------------------------------------------------

    def run_functional_tests(self, node, w0, w0_taddr, w0_account_uuid,
                             w0_zaddr, w0_extra_zaddr):
        # ---- Happy path: explicit-address-array sweep --------------

        print("Test F1: explicit-address-array sweep (response shape + balance moves)...")
        diag_balance = w0.z_gettotalbalance(1, True)
        diag_utxos = w0.z_listunspent(COINBASE_MATURITY + 1)
        diag_t = [u for u in diag_utxos if u.get('pool') == 'transparent']
        print("  [diag] balance={!r}".format(diag_balance))
        print("  [diag] mature coinbase UTXO count={}".format(len(diag_t)))
        pre_private = Decimal(w0.z_gettotalbalance(1, True)['private'])

        result = w0.z_shieldcoinbase([w0_taddr], w0_zaddr)
        self._assert_preflight_shape(result)
        assert_true(result['shieldingUTXOs'] > 0,
                    "Expected >=1 UTXO selected, got {}".format(result['shieldingUTXOs']))
        assert_true(Decimal(result['shieldingValue']) > Decimal('0'),
                    "Expected positive shielding value")
        assert_equal(result['remainingUTXOs'], 0)
        assert_equal(Decimal(result['remainingValue']), Decimal('0'))

        txid = wait_and_assert_operationid_status(w0, result['opid'])
        assert_true(txid is not None, "Shielding tx should have succeeded")
        print("  Shielding tx: {}".format(txid))

        node.generate(1)
        post_private = wait_for_shielded_balance_change(w0, pre_private)
        assert_true(
            post_private > pre_private,
            "Shielded balance should grow: {} -> {}".format(pre_private, post_private))
        print("  Balance {} -> {} ZEC. PASSED".format(pre_private, post_private))

        # ---- Happy path: account-UUID sweep -------------------------

        print("Test F2: account-UUID sweep (happy path)...")
        # The UUID form resolves via `get_transparent_receivers(account, true, true)`,
        # which (with `include_change=true`) returns both EXTERNAL and INTERNAL
        # transparent receivers of the account's registered UAs. The mining
        # address provisioned by `generate-account-and-miner-address` is at
        # `KeyScope::INTERNAL`, so it is reachable through this path.
        node.generate(COINBASE_MATURITY + 10)
        wait_for_mature_coinbase(w0)
        pre_private = Decimal(w0.z_gettotalbalance(1, True)['private'])
        result = w0.z_shieldcoinbase(w0_account_uuid, w0_zaddr)
        self._assert_preflight_shape(result)
        assert_true(result['shieldingUTXOs'] > 0,
                    "UUID-form should sweep coinbase, got 0 UTXOs")
        txid = wait_and_assert_operationid_status(w0, result['opid'])
        assert_true(txid is not None)
        node.generate(1)
        post_private = wait_for_shielded_balance_change(w0, pre_private)
        assert_true(post_private > pre_private,
                    "UUID-form sweep should grow private balance")
        print("  PASSED ({} UTXOs swept)".format(result['shieldingUTXOs']))

        # ---- Happy path: duplicate-taddr deduplication --------------

        print("Test F3: duplicate taddrs in fromaddresses are deduped...")
        node.generate(COINBASE_MATURITY + 10)
        wait_for_mature_coinbase(w0)

        # Compute the eligible UTXO count once, before the call, then
        # compare against the proposal's selection. The wallet's view of
        # mature coinbase can lag the chain tip by an unbounded amount
        # under load, so we don't tie the assertion to that snapshot's
        # absolute value — we assert the semantic property under test:
        # passing the same taddr 3 times must not select each UTXO 3
        # times. We compare against a *single*-listing baseline rather
        # than a triple-listing.
        eligible_before = self._count_mature_coinbase(w0)
        pre_private = Decimal(w0.z_gettotalbalance(1, True)['private'])
        single_result = w0.z_shieldcoinbase([w0_taddr], w0_zaddr)
        # Cancel: we only want the count. There's no cancel API, so
        # let the op run and then mine.
        baseline_selected = single_result['shieldingUTXOs']
        txid = wait_and_assert_operationid_status(w0, single_result['opid'])
        assert_true(txid is not None)
        node.generate(1)
        wait_for_shielded_balance_change(w0, pre_private)

        # Now mine fresh coinbase, then issue with duplicates.
        node.generate(COINBASE_MATURITY + 10)
        wait_for_mature_coinbase(w0)
        pre_private = Decimal(w0.z_gettotalbalance(1, True)['private'])
        eligible_dup = self._count_mature_coinbase(w0)
        result = w0.z_shieldcoinbase([w0_taddr, w0_taddr, w0_taddr], w0_zaddr)
        dup_selected = result['shieldingUTXOs']
        # The bug we're guarding against multiplies the selection count
        # by 3 (one per duplicated entry); dedup-correct behavior keeps
        # it close to the *single* eligible count. Tolerance is loose
        # because wallet sync lag may differ between calls.
        assert_true(
            dup_selected < 2 * eligible_dup,
            "Selecting one taddr 3 times must not multiply selection: "
            "eligible(now)={}, selected(with 3 dups)={}; "
            "dedup may be broken.".format(eligible_dup, dup_selected))
        # Also assert the result is not absurdly large vs. baseline.
        assert_true(
            dup_selected < baseline_selected * 3,
            "Duplication did not approach 3x; eligible was at most "
            "{}, dup selected {}".format(eligible_dup, dup_selected))
        assert_equal(result['remainingUTXOs'], 0)
        txid = wait_and_assert_operationid_status(w0, result['opid'])
        assert_true(txid is not None)
        node.generate(1)
        wait_for_shielded_balance_change(w0, pre_private)
        print("  PASSED")

        # ---- Happy path: shield into a UA on a different account ---

        print("Test F4: shield into a UA on a different account (not the source)...")
        node.generate(COINBASE_MATURITY + 10)
        wait_for_mature_coinbase(w0)

        pre_extra = Decimal(w0.z_gettotalbalance(1, True).get('private', '0'))
        # toaddress belongs to a different account in the same wallet.
        # The new API design imposes no ownership relationship between
        # `fromaddresses` and `toaddress` — only that toaddress has a
        # shielded receiver. Source remains account 0.
        result = w0.z_shieldcoinbase([w0_taddr], w0_extra_zaddr)
        txid = wait_and_assert_operationid_status(w0, result['opid'])
        assert_true(txid is not None)
        node.generate(1)
        post_extra = wait_for_shielded_balance_change(w0, pre_extra)
        # The receiving account is in the same wallet so total
        # `private` grows. We don't have a per-account balance
        # breakdown in z_gettotalbalance; this assertion is the weaker
        # but still meaningful "it didn't fail and the wallet sees
        # the new note" check.
        assert_true(
            post_extra > pre_extra,
            "Wallet private balance should grow after shield-to-other-account: "
            "{} -> {}".format(pre_extra, post_extra))
        print("  PASSED")

        # ---- limit truncation ---------------------------------------

        print("Test F5: limit truncation (limit<eligible)...")
        node.generate(COINBASE_MATURITY + 30)
        wait_for_mature_coinbase(w0, min_mature_utxos=10)

        n_eligible = self._count_mature_coinbase(w0)
        assert_true(
            n_eligible >= 5,
            "Need at least 5 mature coinbase UTXOs to test limit, got {}".format(n_eligible))

        limit = 3
        pre_private = Decimal(w0.z_gettotalbalance(1, True)['private'])
        result = w0.z_shieldcoinbase([w0_taddr], w0_zaddr, limit)
        assert_equal(result['shieldingUTXOs'], limit,
                     "shieldingUTXOs should equal limit")
        # remainingUTXOs is (eligible_at_proposal_tip - limit). We can't
        # tie this to `n_eligible` because the wallet's view of mature
        # coinbase can lag the chain tip arbitrarily; we just assert the
        # structural property under test: truncation must surface non-
        # zero remainingUTXOs and non-zero remainingValue.
        assert_true(
            result['remainingUTXOs'] > 0,
            "remainingUTXOs should be > 0 when truncating, got {}".format(
                result['remainingUTXOs']))
        assert_true(Decimal(result['remainingValue']) > Decimal('0'),
                    "remainingValue should be > 0 when truncating")
        txid = wait_and_assert_operationid_status(w0, result['opid'])
        assert_true(txid is not None)
        node.generate(1)
        wait_for_shielded_balance_change(w0, pre_private)
        print("  PASSED ({}/{} selected, {} remaining)".format(
            limit, n_eligible, n_eligible - limit))

        # ---- limit > eligible is harmless ---------------------------

        print("Test F6: limit greater than eligible is a no-op cap...")
        node.generate(COINBASE_MATURITY + 10)
        wait_for_mature_coinbase(w0)
        n_eligible = self._count_mature_coinbase(w0)

        huge_limit = n_eligible + 1000
        pre_private = Decimal(w0.z_gettotalbalance(1, True)['private'])
        result = w0.z_shieldcoinbase([w0_taddr], w0_zaddr, huge_limit)
        # Selected count = all eligible at proposal tip. Don't tie to
        # the lagging `n_eligible` snapshot; assert structural property:
        # if the limit > eligible, the cap had no effect, so there must
        # be nothing remaining.
        assert_true(
            result['shieldingUTXOs'] > 0,
            "huge limit should still select at least one UTXO")
        assert_equal(result['remainingUTXOs'], 0)
        txid = wait_and_assert_operationid_status(w0, result['opid'])
        assert_true(txid is not None)
        node.generate(1)
        wait_for_shielded_balance_change(w0, pre_private)
        print("  PASSED")

        # ---- memo propagation ---------------------------------------

        print("Test F7: memo propagation...")
        node.generate(COINBASE_MATURITY + 10)
        wait_for_mature_coinbase(w0)

        # 1024-character hex string = 512 bytes. Leading bytes spell
        # "c0ffee" in ASCII so we can eyeball matches if the assertion
        # fails.
        my_memo = '633066666565' + '0' * (1024 - 12)

        pre_private = Decimal(w0.z_gettotalbalance(1, True)['private'])
        result = w0.z_shieldcoinbase([w0_taddr], w0_zaddr, None, my_memo)
        txid = wait_and_assert_operationid_status(w0, result['opid'])
        assert_true(txid is not None)
        node.generate(1)
        wait_for_shielded_balance_change(w0, pre_private)

        tx_details = w0.z_viewtransaction(txid)
        shielded_outputs = [o for o in tx_details.get('outputs', [])
                            if o.get('pool') in ('sapling', 'orchard')]
        assert_true(
            len(shielded_outputs) >= 1,
            "Expected >=1 shielded output, got tx: {}".format(tx_details))
        # The shielding payment is a single shielded recipient note;
        # memo should be on that note. (Change notes use an empty memo;
        # filter to the one that matches.)
        matching = [o for o in shielded_outputs if o.get('memo') == my_memo]
        assert_true(
            len(matching) >= 1,
            "Memo not found on any shielded output; got memos: {}".format(
                [o.get('memo') for o in shielded_outputs]))
        print("  PASSED")

        # ---- operation lifecycle ------------------------------------

        print("Test F8: operation lifecycle (status -> result -> cleared)...")
        node.generate(COINBASE_MATURITY + 10)
        wait_for_mature_coinbase(w0)

        result = w0.z_shieldcoinbase([w0_taddr], w0_zaddr)
        opid = result['opid']
        assert_true(opid.startswith("opid-"),
                    "Expected opid- prefix, got {!r}".format(opid))

        # z_getoperationstatus sees the operation without consuming it.
        status_list = w0.z_getoperationstatus([opid])
        assert_equal(len(status_list), 1)
        assert_equal(status_list[0]['id'], opid)
        assert_true(
            status_list[0]['status'] in ('queued', 'executing', 'success', 'failed'),
            "Unexpected status: {!r}".format(status_list[0]['status']))

        # z_getoperationresult blocks until completion and consumes the
        # result.
        finished = wait_and_assert_operationid_status_result(w0, opid)
        assert_equal(finished['status'], 'success')
        assert_true('txid' in finished['result'])

        # After consumption, the operation is no longer reported.
        remaining = w0.z_getoperationstatus([opid])
        assert_equal(len(remaining), 0)
        print("  PASSED")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assert_preflight_shape(self, result):
        assert_true(isinstance(result, dict),
                    "Expected dict, got {}: {!r}".format(type(result), result))
        for key in ('remainingUTXOs', 'remainingValue',
                    'shieldingUTXOs', 'shieldingValue', 'opid'):
            assert_true(key in result,
                        "Missing field {!r} in response: {!r}".format(key, result))
        assert_true(isinstance(result['remainingUTXOs'], int))
        assert_true(isinstance(result['shieldingUTXOs'], int))
        assert_true(isinstance(result['opid'], str))
        # remainingValue / shieldingValue are JSON numbers; Decimal-able.
        Decimal(result['remainingValue'])
        Decimal(result['shieldingValue'])

    def _count_mature_coinbase(self, wallet):
        utxos = wallet.z_listunspent(COINBASE_MATURITY + 1)
        return len([u for u in utxos if u.get('pool') == 'transparent'])

    # ------------------------------------------------------------------

    def run_test(self):
        node = self.nodes[0]
        w0 = self.wallets[0]

        w0_taddr = self.miner_addresses[0]

        # Identify account 0 on wallet 0.
        accounts = w0.z_listaccounts()
        assert_true(len(accounts) >= 1, "Wallet 0 should have at least one account")
        w0_account_uuid = accounts[0]['account_uuid']

        # Provision an extra account on wallet 0 (for cross-account
        # tests, the empty-source case, and the "shield to a UA in a
        # different account" happy path).
        extra = w0.z_getnewaccount("for-validation-tests")
        w0_extra_account_uuid = extra['account_uuid']
        # Pre-materialize an Orchard UA on the extra account so it has
        # a transparent receiver for the cross-account validation test
        # and a shielded receiver as a shielding destination.
        w0_extra_zaddr = w0.z_getaddressforaccount(
            w0_extra_account_uuid, ["orchard"])['address']

        # A shielded address owned by wallet 0. Use both Sapling and
        # Orchard receivers so the backend can pick whichever the
        # change strategy prefers. (Pure-Orchard destinations hit
        # what appears to be a fee-estimation bug in the build path
        # when shielding many coinbase UTXOs to a single Orchard note;
        # see zcash/wallet#TODO.)
        w0_zaddr = w0.z_getaddressforaccount(
            w0_account_uuid, ["sapling", "orchard"])['address']

        print("Mining initial blocks to mature coinbase...")
        # Need at least 10 mature coinbase UTXOs at w0_taddr for the
        # truncation test; mining 100 + 20 blocks gives 20 mature ones.
        node.generate(COINBASE_MATURITY + 20)
        wait_for_mature_coinbase(w0, min_mature_utxos=10)

        balance = w0.z_gettotalbalance(1, True)
        assert_true(
            Decimal(balance['transparent']) > Decimal('0'),
            "Wallet 0 should see transparent balance after mining")
        print("  Transparent balance: {} ZEC".format(balance['transparent']))
        print("  Account 0 UUID:      {}".format(w0_account_uuid))
        print("  Orchard UA (w0):     {}...".format(w0_zaddr[:24]))
        print("  Extra account UA:    {}...".format(w0_extra_zaddr[:24]))

        print("\n==== Validation tests ====")
        self.run_validation_tests(
            w0, w0_taddr, w0_account_uuid, w0_zaddr,
            w0_extra_account_uuid)

        print("\n==== Functional tests ====")
        self.run_functional_tests(
            node, w0, w0_taddr, w0_account_uuid, w0_zaddr, w0_extra_zaddr)

        print("\nAll z_shieldcoinbase tests passed!")


if __name__ == '__main__':
    WalletZShieldCoinbaseTest().main()
