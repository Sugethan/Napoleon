"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
from chess.engine import PlayResult
from lib.engine_wrapper import MinimalEngine
from lib.types import MOVE, HOMEMADE_ARGS_TYPE
import logging
import copy as copy
from engines.Napoleon import get_move
from engines.Napoleon import node
import keras
import copy


# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)
memory = None

class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

    pass


class Napoleon(ExampleEngine):

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  

        if len(list(board.legal_moves)) == 1:
            return PlayResult(list(board.legal_moves)[0], None)
        else:
            depth = 1000
            exploration = 0.05
            napoleon = keras.saving.load_model('Napoleon.keras')
            global memory
            if memory != None:
                for son in memory.sons:
                    if son.fen == board.fen:
                        memory = son
                        memory.father = None
                        break
            return PlayResult(get_move(napoleon, board, depth, exploration, memory), None)
