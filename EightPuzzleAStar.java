import java.util.*;

class PuzzleState implements Comparable<PuzzleState> {
    int[] board;
    PuzzleState parent;
    String move;
    int depth;
    int cost;

    PuzzleState(int[] b, PuzzleState p, String m, int d, int c) {
        board = b.clone();
        parent = p;
        move = m;
        depth = d;
        cost = c;
    }

    @Override
    public int compareTo(PuzzleState other) {
        return Integer.compare(this.cost, other.cost);
    }
}

public class EightPuzzleAStar {

    static int[] goalState = {1, 2, 3, 4, 5, 6, 7, 8, 0};

    static Map<String, Integer> moves = Map.of(
            "U", -3,
            "D", 3,
            "L", -1,
            "R", 1
    );

    // Heuristic: number of misplaced tiles (excluding blank)
    static int heuristic(int[] board) {
        int count = 0;
        for (int i = 0; i < 9; i++)
            if (board[i] != 0 && board[i] != goalState[i]) count++;
        return count;
    }

    // Returns new board after performing move
    static int[] moveTile(int[] board, String move, int blankPos) {
        int[] newBoard = board.clone();
        int newBlank = blankPos + moves.get(move);

        int temp = newBoard[blankPos];
        newBoard[blankPos] = newBoard[newBlank];
        newBoard[newBlank] = temp;

        return newBoard;
    }

    // A* Search Algorithm
    static PuzzleState aStar(int[] start) {
        PriorityQueue<PuzzleState> openList = new PriorityQueue<>();
        Set<String> closedList = new HashSet<>();

        openList.add(new PuzzleState(start, null, null, 0, heuristic(start)));

        while (!openList.isEmpty()) {
            PuzzleState current = openList.poll();

            if (Arrays.equals(current.board, goalState))
                return current;

            closedList.add(Arrays.toString(current.board));

            int blankPos = -1;
            for (int i = 0; i < 9; i++)
                if (current.board[i] == 0) blankPos = i;

            for (String move : moves.keySet()) {
                // Check invalid moves
                if (move.equals("U") && blankPos < 3) continue;
                if (move.equals("D") && blankPos > 5) continue;
                if (move.equals("L") && blankPos % 3 == 0) continue;
                if (move.equals("R") && blankPos % 3 == 2) continue;

                int[] newBoard = moveTile(current.board, move, blankPos);

                if (closedList.contains(Arrays.toString(newBoard))) continue;

                PuzzleState newState = new PuzzleState(
                        newBoard,
                        current,
                        move,
                        current.depth + 1,
                        current.depth + 1 + heuristic(newBoard)
                );

                openList.add(newState);
            }
        }
        return null;
    }

    // Print board
    static void printBoard(int[] board) {
        System.out.println("+---+---+---+");
        for (int i = 0; i < 9; i += 3) {
            System.out.printf("| %s | %s | %s |%n",
                    board[i] == 0 ? " " : board[i],
                    board[i + 1] == 0 ? " " : board[i + 1],
                    board[i + 2] == 0 ? " " : board[i + 2]
            );
            System.out.println("+---+---+---+");
        }
    }

    // Print solution path
    static void printSolution(PuzzleState solution) {
        List<PuzzleState> path = new ArrayList<>();
        PuzzleState current = solution;

        while (current != null) {
            path.add(current);
            current = current.parent;
        }

        Collections.reverse(path);

        for (PuzzleState step : path) {
            System.out.println(step.move == null ? "Initial State:" : "Move: " + step.move);
            printBoard(step.board);
        }

        System.out.println("Reached Goal State!");
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[] startState = new int[9];

        System.out.println("Enter initial state (row by row, use 0 for blank):");
        for (int i = 0; i < 9; i++) {
            startState[i] = sc.nextInt();
        }

        PuzzleState solution = aStar(startState);

        if (solution != null) {
            System.out.println("\nSolution found!");
            printSolution(solution);
        } else {
            System.out.println("No solution exists.");
        }
    }
}
