import asyncio
import concurrent
import concurrent.futures
import types
from functools import partial, update_wrapper
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import ParamSpec

from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START, TAG_HIDDEN
from langgraph.pregel import Pregel
from langgraph.pregel.call import get_runnable_for_func
from langgraph.pregel.read import PregelNode
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

P = ParamSpec("P")
P1 = TypeVar("P1")
T = TypeVar("T")


def call(
    func: Callable[[P1], T],
    input: P1,
    *,
    retry: Optional[RetryPolicy] = None,
) -> concurrent.futures.Future[T]:
    from langgraph.constants import CONFIG_KEY_CALL
    from langgraph.utils.config import get_configurable

    conf = get_configurable()
    impl = conf[CONFIG_KEY_CALL]
    fut = impl(func, input, retry=retry)
    return fut


@overload
def task(
    *, retry: Optional[RetryPolicy] = None
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, asyncio.Future[T]]]: ...


@overload
def task(  # type: ignore[overload-cannot-match]
    *, retry: Optional[RetryPolicy] = None
) -> Callable[[Callable[P, T]], Callable[P, concurrent.futures.Future[T]]]: ...


def task(
    *, retry: Optional[RetryPolicy] = None
) -> Union[
    Callable[[Callable[P, Awaitable[T]]], Callable[P, asyncio.Future[T]]],
    Callable[[Callable[P, T]], Callable[P, concurrent.futures.Future[T]]],
]:
    def _task(func: Callable[P, T]) -> Callable[P, concurrent.futures.Future[T]]:
        return update_wrapper(partial(call, func, retry=retry), func)

    return _task


def entrypoint(
    *,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
) -> Callable[[types.FunctionType], Pregel]:
    def _imp(func: types.FunctionType) -> Pregel:
        return Pregel(
            nodes={
                func.__name__: PregelNode(
                    bound=get_runnable_for_func(func),
                    triggers=[START],
                    channels=[START],
                    writers=[ChannelWrite([ChannelWriteEntry(END)], tags=[TAG_HIDDEN])],
                )
            },
            channels={START: EphemeralValue(Any), END: LastValue(Any, END)},
            input_channels=START,
            output_channels=END,
            stream_channels=END,
            stream_mode="updates",
            checkpointer=checkpointer,
            store=store,
        )

    return _imp
