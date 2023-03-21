import random
import sys
import copy
from tabnanny import verbose
import time
from host import GO
from read import readInput
from write import writeOutput

from go_play import GOPlay


class MyPlayer():
    def __init__(self):
       self=self


    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''        
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_move(i, j, piece_type):
                    possible_placements.append((i,j))

        if not possible_placements:
            return "PASS"
        else:
            return random.choice(possible_placements)

    def findLocations(self,board, piece_type):
        placement = []
        for i in range(go.size):
            for j in range(go.size):
                if board[i][j] == piece_type:
                    placement.append((i,j))
        return placement

    def alphaBetaPruningMin(self,board, color, depth, alpha, beta, startTime):
        newBoard = copy.deepcopy(board)

        currentMin = float('inf')
        moves = go.empty_cells(color)

        end = time.time()
        if len(moves) == 0 or depth == 0 or end - startTime> 8.5:
            return (-1,-1), go.evaluate(color)
        else: 
            for move in moves:

                boardEachIteration = copy.deepcopy(board)
                newBoard =  go.set_move(boardEachIteration, move[0], move[1], color)
                go.remove_died_pieces(3 - color)
                if color == 1:
                    nextPlayer = 2
                else:
                    nextPlayer = 1
                newMove, newScore = self.alphaBetaPruningMax(newBoard, nextPlayer, depth - 1, alpha, beta, startTime)
                if newScore < currentMin:
                    currentMin = newScore
                    bestMove = move
                beta = min(newScore, beta) 
                if beta <= alpha:
                    break
            return bestMove, currentMin 

    def alphaBetaPruningMax(self,board, color, depth, alpha, beta, startTime):
        end = time.time()
        newBoard = copy.deepcopy(board)
        currentMax = float('-inf')
        moves = go.empty_cells(color)

        stonesToRemove = []
        for move in moves:
            go.board[move[0]][move[1]] = color
            oppositeMoves = go.empty_cells(3 - color)
            for oMove in oppositeMoves:
                go.board[oMove[0]][oMove[1]] = 3 - color
                deadstones = go.find_died_pieces(color)
                go.board[oMove[0]][oMove[1]] = 0
                if move in deadstones:
                    if move not in stonesToRemove:
                        stonesToRemove.append(move)
            go.board[move[0]][move[1]] = 0

        for stone in stonesToRemove:
            if stone in moves:
                moves.remove(stone)
        
        if len(moves) == 0 or depth == 0 or end - startTime> 8.5:
            return (-1,-1), go.evaluate(color)
        else: 
            for move in moves:

                boardEachIteration = copy.deepcopy(board)
                newBoard = go.set_move(boardEachIteration, move[0], move[1], color)
                go.remove_died_pieces(3 - color)
                if color == 1:
                    nextPlayer = 2
                else:
                    nextPlayer = 1
                newMove, newScore = self.alphaBetaPruningMin(newBoard, nextPlayer, depth - 1, alpha, beta, startTime)
                if newScore > currentMax:
                    currentMax = newScore
                    bestMove = move
                alpha = max(newScore, alpha) 
                if beta <= alpha:
                    break
            return bestMove, currentMax
    

    def selectMoveMinMax(self,board, color):
        start = time.time()
        bestMove, score = self.alphaBetaPruningMax(board, color, max_depth, float('-inf'),  float('inf'), start)
        return bestMove[0],bestMove[1], score
    
    def goPlayAlphaBetaPruningMax(self,goPlay, board, color, depth, alpha, beta, startTime):
        end = time.time()
        newBoard = copy.deepcopy(board)
        currentMax = float('-inf')
        moves = goPlay.empty_cells(color)

        if len(moves) == 0 or depth == 0 or end - startTime> 8.5:
                return (-1,-1), goPlay.evaluate(color)
        else: 
            for move in moves:

                boardEachIteration = copy.deepcopy(board)
                newBoard = goPlay.set_move(boardEachIteration, move[0], move[1], color)
                goPlay.remove_died_pieces(3 - color)
                if color == 1:
                    nextPlayer = 2
                else:
                    nextPlayer = 1
                newMove, newScore = self.goPlayAlphaBetaPruningMin(goPlay,newBoard, nextPlayer, depth - 1, alpha, beta, startTime)
                if newScore > currentMax:
                    currentMax = newScore
                    bestMove = move
                alpha = max(newScore, alpha) 
                if beta <= alpha:
                    break
            return bestMove, currentMax
    
    def goPlayAlphaBetaPruningMin(self,goPlay,board, color, depth, alpha, beta, startTime):
        newBoard = copy.deepcopy(board)

        currentMin = float('inf')
        moves = goPlay.empty_cells(color)

        end = time.time()
        if len(moves) == 0 or depth == 0 or end - startTime> 8.5:
            return (-1,-1), goPlay.evaluate(color)
        else: 
            for move in moves:

                boardEachIteration = copy.deepcopy(board)
                newBoard =  goPlay.set_move(boardEachIteration, move[0], move[1], color)
                goPlay.remove_died_pieces(3 - color)
                if color == 1:
                    nextPlayer = 2
                else:
                    nextPlayer = 1
                newMove, newScore = self.goPlayAlphaBetaPruningMax(goPlay,newBoard, nextPlayer, depth - 1, alpha, beta, startTime)
                if newScore < currentMin:
                    currentMin = newScore
                    bestMove = move
                beta = min(newScore, beta) 
                if beta <= alpha:
                    break
            return bestMove, currentMin 

    def goPlaySelectMoveMinMax(self,goPlay, board, color):
        start = time.time()
        bestMove, score = self.goPlayAlphaBetaPruningMax(goPlay, board, color, max_depth_opponent, float('-inf'),  float('inf'), start )
        return bestMove[0], bestMove[1], score
        
    def getInputSmart(self,go,piece_type):
        remainingSpaces=[]
        boardSize=go.size
        killedPiecesCount=dict()
       
        movesToRemove=[]
        movesToSave=dict()
        opponentMoves=[]
        emptyX=[]
        neighbours=[]
        for i in range(boardSize):
            for j in range(boardSize):
                if go.board[i][j]==0:
                    remainingSpaces.append((i,j))
        
        for space in remainingSpaces:
            go.board[space[0]][space[1]]=piece_type
            deadPieces=go.find_died_pieces(3-piece_type)
            go.board[space[0]][space[1]]=0
            if len(deadPieces) >= 1:
                killedPiecesCount[space]=len(deadPieces)
        
        sortedKilledPiecesCount=sorted(killedPiecesCount,key=killedPiecesCount.get,reverse=True)
        
        for kill in sortedKilledPiecesCount:
            sampleBoard=copy.deepcopy(go.board)
            sampleBoard[kill[0]][kill[1]]=piece_type
            deadStones=go.find_died_pieces_moves(sampleBoard, 3 - piece_type)
            for stone in deadStones:
                sampleBoard[stone[0]][stone[1]]=0
            if kill!=None and go.previous_board != sampleBoard:
                return kill

        boardMoves=go.empty_cells(piece_type)

        for move in boardMoves:
            go.board[move[0]][move[1]] = piece_type
            oppMoves = go.empty_cells_moves(go.board, 3-piece_type)
            for oppMove in oppMoves:
                go.board[oppMove[0]][oppMove[1]] = 3 - piece_type
                deadPieces = go.find_died_pieces(piece_type)
                go.board[oppMove[0]][oppMove[1]] = 0
                if move in deadPieces:
                    movesToRemove.append(move)
            go.board[move[0]][move[1]] = 0
        
        for move in movesToRemove:
            if move in boardMoves:
                boardMoves.remove(move)
        

        if len(boardMoves)==0:
            return "PASS"

    
        for i in range(boardSize):
            for j in range(boardSize):
                if go.board[i][j] == 0:
                    opponentMoves.append((i,j))

        for move in opponentMoves:
            go.board[move[0]][move[1]] = 3-piece_type
            playersDeadStones = go.find_died_pieces(piece_type)
            go.board[move[0]][move[1]] = 0
            if len(playersDeadStones) >=1:
                movesToSave[move] = len(playersDeadStones)

        sortedSavedMoves = sorted(movesToSave, key = movesToSave.get, reverse = True)

        for move in sortedSavedMoves:
            if move != None and move in boardMoves:
                return move

        positionOfOpponent = self.findLocations(go.board, 3-piece_type)

        for position in positionOfOpponent:
            neighbors = [(position[0]+board[0], position[1]+board[1]) for board in 
                    [(-1,0), (1,0), (0,-1), (0,1)] 
                    if ( (0 <= position[0]+board[0] < boardSize) and (0 <= position[1]+board[1] < boardSize))]
            for neighbor in neighbors:
                neighbours.append(neighbor)


        for neighbour in neighbours:
            if board[neighbour[0]][neighbour[1]]==0:
                emptyX.append(neighbour)
        
        for move in boardMoves:
            sampleBoard=copy.deepcopy(go.board)
            sampleBoard[move[0]][move[1]] = piece_type
            deadStones = go.find_died_pieces_moves(sampleBoard, 3 - piece_type)
            for stone in deadStones:
                sampleBoard[stone[0]][stone[1]] = 0
            positionOfOpponent = self.findLocations(sampleBoard, 3 - piece_type)
            emptyY = []
            neighbours = []

            for position in positionOfOpponent:
                neighbors = [(position[0]+board[0], position[1]+board[1]) for board in 
                            [(-1,0), (1,0), (0,-1), (0,1)] 
                            if ( (0 <= position[0]+board[0] < boardSize) and (0 <= position[1]+board[1] < boardSize))]
                for neighbor in neighbors:
                    neighbours.append(neighbor)

            for neighbour in neighbours:
                if board[neighbour[0]][neighbour[1]] == 0:
                    emptyY.append(neighbour)

            if len(emptyX) - len(emptyY) >=1:
                return move

        if len(boardMoves)>=15:
            if (2,2) in boardMoves:
                return (2,2)
            if (1,1) in boardMoves:
                return (1,1)
            if (1,3) in boardMoves:
                return (1,3)
            if (3,1) in boardMoves:
                return (3,1)
            if (3,3) in boardMoves:
                return (3,3)
            if (2,0) in boardMoves:
                return (2,0)
            if (2,4) in boardMoves:
                return (2,4)
            if (0,2) in boardMoves:
                return (0,2)
            if (4,2) in boardMoves:
                return (4,2)
        
        opponentsBoard = copy.deepcopy(go.board)
        opponentsPreviousBoard = copy.deepcopy(go.previous_board)
        goPlay = GOPlay(5)
        goPlay.set_board(3-piece_type, opponentsPreviousBoard, opponentsBoard)
        moveI, moveJ, score = self.goPlaySelectMoveMinMax(goPlay, opponentsBoard, 3-piece_type)
        x, y = moveI, moveJ
        go.board[x][y] = 3 - piece_type
        emptySpacesOnBoard = []
        for i in range(boardSize):
            for j in range(boardSize):
                if go.board[i][j] == 0:
                    emptySpacesOnBoard.append((i,j))
        
        killedCount = dict()
        for space in emptySpacesOnBoard:
            go.board[space[0]][space[1]] = piece_type
            deadPieces = go.find_died_pieces(3-piece_type)
            go.board[space[0]][space[1]] = 0
            if len(deadPieces) >= 1:
                killedCount[space] = len(deadPieces)

        killCountRemove = []
        sortedKilledCount = sorted(killedCount, key = killedCount.get, reverse = True)
        go.board[x][y] = 0

        if len(sortedKilledCount) != 0:
            for killCount in sortedKilledCount:
                go.board[killCount[0]][killCount[1]] == piece_type
                oppositeMoves = go.empty_cells_moves(go.board, 3- piece_type)
                for move in oppositeMoves:
                    go.board[move[0]][move[1]] = 3 - piece_type
                    died_pieces = go.find_died_pieces_moves(go.board, piece_type)
                    go.board[move[0]][move[1]] = 0
                    if killCount in died_pieces:
                        killCountRemove.append(i)
                go.board[killCount[0]][killCount[1]] = 0

            for kill in killCountRemove:
                if kill in sortedKilledCount:
                    sortedKilledCount.remove(kill)


            for kill in sortedKilledCount:
                if kill in boardMoves:
                    return kill

        moveI, moveJ, score = self.selectMoveMinMax(go.board, piece_type)
        return (moveI,moveJ)  




if __name__ == "__main__":
    N,max_depth, max_depth_opponent=5,4,1
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type,previous_board,board)
    myPlayer=MyPlayer()
    action=myPlayer.getInputSmart(go,piece_type)
    if action == None:
        action = "PASS"
    writeOutput(action)