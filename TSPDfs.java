import java.util.*;

public class TSPDfs {

    private static class State {
        List<Integer> path;
        boolean[] visited;
        int costSoFar;

        State(List<Integer> path, boolean[] visited, int costSoFar) {
            this.path = path;
            this.visited = visited;
            this.costSoFar = costSoFar;
        }
    }

    private static class Result {
        List<Integer> tour; // includes return to start at the end
        int totalCost;

        Result(List<Integer> tour, int totalCost) {
            this.tour = tour;
            this.totalCost = totalCost;
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of cities (N): ");
        int n = readInt(sc);

        if (n <= 1) {
            System.out.println("N must be >= 2");
            return;
        }

        System.out.println("Enter adjacency matrix (use 0 to indicate no direct edge; diagonal must be 0):");
        int[][] graph = new int[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                graph[i][j] = readInt(sc);
            }
        }

        System.out.print("Enter start city index (0-based): ");
        int start = readInt(sc);

        if (start < 0 || start >= n) {
            System.out.println("Invalid start index");
            return;
        }

        System.out.println("\n=== DFS TSP ===");

        Result dfs = solveDfs(graph, start);

        if (dfs != null) {
            printDfsResult(dfs, graph);
        } else {
            System.out.println("No Hamiltonian cycle found (graph may be disconnected).");
        }
    }

    private static int readInt(Scanner sc) {
        while (true) {
            try {
                return Integer.parseInt(sc.next());
            } catch (Exception e) {
                System.out.print("Please enter an integer: ");
            }
        }
    }

    private static Result solveDfs(int[][] graph, int start) {
        int n = graph.length;

        Deque<State> stack = new ArrayDeque<>();
        boolean[] initVisited = new boolean[n];
        initVisited[start] = true;

        stack.push(new State(new ArrayList<>(List.of(start)), initVisited, 0));

        int bestCost = Integer.MAX_VALUE;
        List<Integer> bestTour = null;

        while (!stack.isEmpty()) {
            State s = stack.pop();

            // If Hamiltonian path found
            if (s.path.size() == n) {
                int last = s.path.get(s.path.size() - 1);
                int back = graph[last][start];

                if (back > 0) {
                    int total = s.costSoFar + back;

                    if (total < bestCost) {
                        bestCost = total;
                        bestTour = new ArrayList<>(s.path);
                        bestTour.add(start);
                    }
                }
                continue;
            }

            if (s.costSoFar >= bestCost) {
                continue; // prune
            }

            int last = s.path.get(s.path.size() - 1);

            // DFS: explore in reverse order to get lexicographically increasing results
            for (int next = n - 1; next >= 0; next--) {
                if (!s.visited[next] && graph[last][next] > 0) {

                    boolean[] nextVisited = Arrays.copyOf(s.visited, n);
                    nextVisited[next] = true;

                    List<Integer> nextPath = new ArrayList<>(s.path);
                    nextPath.add(next);

                    int nextCost = s.costSoFar + graph[last][next];

                    stack.push(new State(nextPath, nextVisited, nextCost));
                }
            }
        }

        return bestTour == null ? null : new Result(bestTour, bestCost);
    }

    private static void printDfsResult(Result result, int[][] graph) {
        System.out.println("Best tour (DFS): " + formatTour(result.tour));
        System.out.println("Total cost: " + result.totalCost);
        System.out.println("Depth-order edge steps:");

        int cumulative = 0;

        for (int i = 0; i < result.tour.size() - 1; i++) {
            int u = result.tour.get(i);
            int v = result.tour.get(i + 1);
            int w = graph[u][v];

            cumulative += w;

            System.out.println(" Depth " + (i + 1) + ": " + u + " -> " + v +
                    " (edge cost=" + w + ", cumulative=" + cumulative + ")");
        }
    }

    private static String formatTour(List<Integer> tour) {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < tour.size(); i++) {
            if (i > 0) sb.append(" -> ");
            sb.append(tour.get(i));
        }

        return sb.toString();
    }
}
