from typing import Any, Callable, List


def batched(
    op: Callable[..., Any],
    batch_size: int,
    *args: Any,
    **kwargs: Any,
) -> List[Any]:
    """Batched operation.

    Args:
        op: Operation to be batched.
        batch_size: Batch size.
        *args: Arguments for the operation.
        **kwargs: Keyword arguments for the operation.

    Returns:
        List of results.
    """
    results = []
    for i in range(0, len(args[0]), batch_size):
        results.extend(op(*[arg[i : i + batch_size] for arg in args], **kwargs))
    return results


def ddp_batched(
    op: Callable[..., Any],
    batch_size: int,
    world_size: int,
    *args: Any,
    **kwargs: Any,
) -> List[Any]:
    """Batched operation with DistributedDataParallel.

    Args:
        op: Operation to be batched.
        batch_size: Batch size.
        *args: Arguments for the operation.
        **kwargs: Keyword arguments for the operation.

    Returns:
        List of results.
    """
    results = []
    for i in range(0, len(args[0]), batch_size):
        results.extend(op(*[arg[i : i + batch_size] for arg in args], **kwargs))
    return results
