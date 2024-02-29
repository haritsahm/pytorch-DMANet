import logging
from typing import Mapping, Optional

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        Parameters
        ----------
        name : str, optional
            The name of the logger, by default __name__
        rank_zero_only : bool, optional
            Whether to force all logs to only occur on the rank zero process, by default False
        extra : Optional[Mapping[str, object]], optional
            A dict-like object which provides contextual information. See `logging.LoggerAdapter`,
            by default None
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only occur
        on that rank/process.

        Parameters
        ----------
        level : int
            The level to log at. Look at `logging.__init__.py` for more information.
        msg : str
            The message to log.
        rank : Optional[int], optional
            The rank to log at, by default None

        Raises
        ------
        RuntimeError
            rank_zero_only is not initialized
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, 'rank', None)
            if current_rank is None:
                raise RuntimeError('The `rank_zero_only.rank` needs to be set before use')
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
