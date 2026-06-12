"""
TwoTierStore — a BaseStore wrapper that pairs a warm tier (typically
WarmStore) with a durable backing tier (typically PostgresStore,
InMemoryStore with an embedding index, or any other BaseStore).

The agent code uses TwoTierStore as if it were a single BaseStore. Reads
try warm first and fall back to durable on miss; writes mirror to both,
with the async path firing both writes concurrently via `asyncio.gather`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable, Sequence

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
)

log = logging.getLogger(__name__)


class TwoTierStore(BaseStore):
    """
    LangGraph BaseStore that wraps a warm tier and a durable tier.

    Reads consult the warm tier first; on a miss (no results, or top score
    below `warm_hit_threshold`), reads fall through to the durable tier and
    populate the warm tier with the results on the way back.

    Writes mirror to both tiers (write-through). The async path
    (`aput`, `abatch`) fires both writes concurrently with `asyncio.gather`,
    so total write latency is `max(warm_put, durable_put)` instead of the
    sum. The sync path (`put`, `batch`) writes sequentially — the savings
    from sync-parallelizing the warm/durable writes are too small
    (microseconds) to justify a thread-pool lifecycle.

    Error semantics: **durable success is required**. If the durable write
    raises, the warm write is cancelled (in the async path) and the error
    propagates. Warm errors are tolerated by default (`fail_on_warm_error=False`)
    because the durable tier already has the data — the cache will rehydrate
    on the next read miss.

    Example:
        from langgraph.store.memory import InMemoryStore
        from warm_memory.langgraph import WarmStore, TwoTierStore

        warm = WarmStore(capacity=32)
        durable = InMemoryStore(index={"embed": my_embeddings, "dims": 1536, "fields": ["text"]})
        store = TwoTierStore(warm=warm, durable=durable, warm_hit_threshold=0.34)

        graph = StateGraph(...).compile(store=store)
        # Agent code uses `store` like any single BaseStore — TwoTierStore handles the rest.
    """

    __slots__ = (
        "warm",
        "durable",
        "warm_hit_threshold",
        "populate_warm_on_miss",
        "write_through",
        "fail_on_warm_error",
    )

    def __init__(
        self,
        warm: BaseStore,
        durable: BaseStore,
        *,
        warm_hit_threshold: float = 0.34,
        populate_warm_on_miss: bool = True,
        write_through: bool = True,
        fail_on_warm_error: bool = False,
    ) -> None:
        self.warm = warm
        self.durable = durable
        self.warm_hit_threshold = warm_hit_threshold
        self.populate_warm_on_miss = populate_warm_on_miss
        self.write_through = write_through
        self.fail_on_warm_error = fail_on_warm_error

    # ---------------------------------------------------------------- helpers

    def _is_warm_hit(self, results: list[Any] | None) -> bool:
        if not results:
            return False
        top = results[0]
        score = getattr(top, "score", None)
        # If the warm scorer doesn't return scores at all, treat any result
        # as a hit (the threshold only applies when scoring is in play).
        if score is None:
            return True
        return float(score) >= self.warm_hit_threshold

    def _populate_warm(self, namespace: tuple[str, ...], items: Iterable[Any]) -> None:
        """Best-effort: copy durable items into warm. Warm errors are logged."""
        for item in items:
            try:
                self.warm.put(namespace, item.key, dict(item.value))
            except Exception as exc:
                if self.fail_on_warm_error:
                    raise
                log.warning("warm populate failed for %s/%s: %s", namespace, item.key, exc)

    async def _apopulate_warm(self, namespace: tuple[str, ...], items: Iterable[Any]) -> None:
        for item in items:
            try:
                await self.warm.aput(namespace, item.key, dict(item.value))
            except Exception as exc:
                if self.fail_on_warm_error:
                    raise
                log.warning("warm populate (async) failed for %s/%s: %s", namespace, item.key, exc)

    # ---------------------------------------------------------------- ops (sync)

    def _do_get_sync(self, op: GetOp) -> Item | None:
        item = self.warm.get(op.namespace, op.key)
        if item is not None:
            return item
        item = self.durable.get(op.namespace, op.key)
        if item is not None and self.populate_warm_on_miss:
            self._populate_warm(op.namespace, [item])
        return item

    def _do_search_sync(self, op: SearchOp) -> list[Any]:
        warm_results = self.warm.search(
            op.namespace_prefix,
            query=op.query,
            filter=op.filter,
            limit=op.limit,
            offset=op.offset,
        )
        if self._is_warm_hit(warm_results):
            return warm_results

        durable_results = self.durable.search(
            op.namespace_prefix,
            query=op.query,
            filter=op.filter,
            limit=op.limit,
            offset=op.offset,
        )
        if self.populate_warm_on_miss and durable_results:
            self._populate_warm(op.namespace_prefix, durable_results)
        return durable_results

    def _do_put_sync(self, op: PutOp) -> None:
        # Sync path: sequential. Use the async API (`aput`/`abatch`) for parallel writes.
        if self.write_through:
            try:
                if op.value is None:
                    self.warm.delete(op.namespace, op.key)
                else:
                    self.warm.put(op.namespace, op.key, op.value, index=op.index)
            except Exception as exc:
                if self.fail_on_warm_error:
                    raise
                log.warning("warm write failed for %s/%s: %s", op.namespace, op.key, exc)

        if op.value is None:
            self.durable.delete(op.namespace, op.key)
        else:
            self.durable.put(op.namespace, op.key, op.value, index=op.index)

    def _do_list_namespaces_sync(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        return self.durable.batch([op])[0]  # forward to durable (ground truth)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        ops_list: Sequence[Op] = list(ops)
        results: list[Result] = []
        for op in ops_list:
            if isinstance(op, GetOp):
                results.append(self._do_get_sync(op))
            elif isinstance(op, PutOp):
                self._do_put_sync(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._do_search_sync(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._do_list_namespaces_sync(op))
            else:
                raise TypeError(f"Unsupported op type: {type(op).__name__}")
        return results

    # ---------------------------------------------------------------- ops (async)

    async def _ado_get(self, op: GetOp) -> Item | None:
        item = await self.warm.aget(op.namespace, op.key)
        if item is not None:
            return item
        item = await self.durable.aget(op.namespace, op.key)
        if item is not None and self.populate_warm_on_miss:
            await self._apopulate_warm(op.namespace, [item])
        return item

    async def _ado_search(self, op: SearchOp) -> list[Any]:
        warm_results = await self.warm.asearch(
            op.namespace_prefix,
            query=op.query,
            filter=op.filter,
            limit=op.limit,
            offset=op.offset,
        )
        if self._is_warm_hit(warm_results):
            return warm_results

        durable_results = await self.durable.asearch(
            op.namespace_prefix,
            query=op.query,
            filter=op.filter,
            limit=op.limit,
            offset=op.offset,
        )
        if self.populate_warm_on_miss and durable_results:
            await self._apopulate_warm(op.namespace_prefix, durable_results)
        return durable_results

    async def _ado_put(self, op: PutOp) -> None:
        """
        The headline async win: warm and durable writes run concurrently via
        asyncio.gather. Total latency is ~max(warm, durable) instead of the
        sum. Durable success is required; warm failure is tolerated by
        default.
        """
        async def _warm_write() -> None:
            if op.value is None:
                await self.warm.adelete(op.namespace, op.key)
            else:
                await self.warm.aput(op.namespace, op.key, op.value, index=op.index)

        async def _durable_write() -> None:
            if op.value is None:
                await self.durable.adelete(op.namespace, op.key)
            else:
                await self.durable.aput(op.namespace, op.key, op.value, index=op.index)

        if not self.write_through:
            await _durable_write()
            return

        warm_task = asyncio.create_task(_warm_write())
        try:
            await _durable_write()
        except Exception:
            warm_task.cancel()
            try:
                await warm_task
            except (asyncio.CancelledError, Exception):
                pass
            raise

        try:
            await warm_task
        except Exception as exc:
            if self.fail_on_warm_error:
                raise
            log.warning("warm write (async) failed for %s/%s: %s", op.namespace, op.key, exc)

    async def _ado_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        return (await self.durable.abatch([op]))[0]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        ops_list = list(ops)
        results: list[Result] = []
        for op in ops_list:
            if isinstance(op, GetOp):
                results.append(await self._ado_get(op))
            elif isinstance(op, PutOp):
                await self._ado_put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(await self._ado_search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(await self._ado_list_namespaces(op))
            else:
                raise TypeError(f"Unsupported op type: {type(op).__name__}")
        return results
