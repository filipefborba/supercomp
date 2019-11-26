#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>
namespace mpi = boost::mpi;

double dist(std::pair<double, double> &p1, std::pair<double, double> &p2)
{
    return sqrt(pow((p1.first - p2.first), 2) + pow((p1.second - p2.second), 2));
}

void calc_dist(std::vector<std::pair<double, double>> &points, std::vector<std::vector<double>> &points_distance)
{
    for (int i = 0; i < points.size(); i++)
    {
        for (int j = 0; j < points.size(); j++)
        {
            points_distance[i][j] = dist(points[i], points[j]);
        }
    }
}

double total_cost(std::vector<int> &solution, std::vector<std::vector<double>> &points_distance, int N)
{
    double solution_cost = points_distance[solution[0]][solution[N - 1]]; // Primeiro calculo: primeiro e ultimo pontos
    for (int k = 1; k < N; k++)
    {
        solution_cost += points_distance[solution[k - 1]][solution[k]]; // Calculo das distancias
    }
    return solution_cost;
}

double backtrack(int idx, double curr_cost, std::vector<int> &curr_sol, double &best_cost, std::vector<int> &best_seq, std::vector<bool> &used, std::vector<std::vector<double>> &points_distance, int N, int id, int n_machines)
{
    if (idx == N)
    {
        curr_cost += points_distance[curr_sol[0]][curr_sol.back()];
        if (curr_cost < best_cost)
        {
            best_seq = curr_sol;
            best_cost = curr_cost;
        }
        return best_cost;
    }

    for (int i = 0; i < N; i++)
    {
        if (used[i] == false)
        {
            used[i] = true;
            curr_sol[idx] = i;

            double new_cost = curr_cost + points_distance[curr_sol[idx - 1]][curr_sol[idx]];

            // No primeiro passo da recursao
            if (idx == 1)
            {
                // Divide o trabalho entre cada maquina
                // Exemplo: para N = 6, a principal faz 1 e 4, cluster1 faz 2 e 5, cluster2 faz 3.
                if (i % n_machines == id)
                {
                    best_cost = backtrack(idx + 1, new_cost, curr_sol, best_cost, best_seq, used, points_distance, N, id, n_machines);
                }
                // Se ele nao passar pelo if, ele pula, sem realizar o trabalho que as outras maquinas estao fazendo
            }
            else
            {
                best_cost = backtrack(idx + 1, new_cost, curr_sol, best_cost, best_seq, used, points_distance, N, id, n_machines);
            }

            used[i] = false;
            curr_sol[idx] = -1;
        }
    }
    return best_cost;
}

int main(int argc, char *argv[])
{
    mpi::environment env(argc, argv);
    mpi::communicator world;
    int id = world.rank();
    int n_machines = world.size();

    if (id == 0)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Preparacao para receber os dados do arquivo
        int N;
        std::cin >> N;
        std::vector<std::pair<double, double>> points(N);
        std::vector<bool> used(N);    // Pontos usados
        std::vector<int> curr_sol(N); // Solucao atual
        std::vector<int> best_sol(N); // Melhor solucao

        double x, y;
        std::pair<double, double> point;
        for (int i = 0; i < N; i++)
        {
            std::cin >> x;
            std::cin >> y;
            point.first = x;
            point.second = y;
            points[i] = point;

            used[i] = false;
            curr_sol[i] = -1;
            best_sol[i] = -1;
        }
        curr_sol[0] = 0;
        used[0] = true;

        // ---------------------------------------------------------------------
        // Preparacao para pre-calcular as distancias
        std::vector<std::vector<double>> points_distance(N, std::vector<double>(N, 0.0));

        calc_dist(points, points_distance);

        // ---------------------------------------------------------------------
        // Broadcast das informacoes para outras maquinas
        mpi::broadcast(world, N, 0);
        mpi::broadcast(world, points_distance, 0);
        mpi::broadcast(world, used, 0);
        mpi::broadcast(world, curr_sol, 0);
        mpi::broadcast(world, best_sol, 0);
        // ---------------------------------------------------------------------
        // Enumeracao exaustiva

        double best_cost = std::numeric_limits<double>::infinity(); // Melhor custo atual

        backtrack(1, 0, curr_sol, best_cost, best_sol, used, points_distance, N, id, n_machines);

        // ---------------------------------------------------------------------
        // Recebe das outras máquinas e acha o melhor caminho

        std::vector<std::vector<int>> best_solutions(n_machines, std::vector<int>(N, 0));
        std::vector<double> best_costs(n_machines);

        mpi::gather(world, best_cost, best_costs, 0);
        mpi::gather(world, best_sol, best_solutions, 0);

        auto iter = std::min_element(best_costs.begin(), best_costs.end());
        int position = std::distance(best_costs.begin(), iter);
        best_cost = *iter;
        best_sol = best_solutions[position];

        auto end_time = std::chrono::high_resolution_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

// ---------------------------------------------------------------------
// Print do tempo e do melhor caminho
#ifdef TIME
        std::cout << runtime.count() << std::endl;
        std::cout << "milisegundo(s)." << std::endl;
#endif

        std::cout << std::fixed << std::setprecision(5);
        std::cout << best_cost;
        std::cout << " 1" << std::endl;

        for (int i = 0; i < N; i++)
        {
            std::cout << best_sol[i] << ' ';
        }
        std::cout << std::endl;
    }
    else
    {
        int recv_N;
        std::vector<std::vector<double>> recv_points_distance;
        std::vector<bool> recv_used;
        std::vector<int> recv_curr_sol;
        std::vector<int> recv_best_sol;

        mpi::broadcast(world, recv_N, 0);
        mpi::broadcast(world, recv_points_distance, 0);
        mpi::broadcast(world, recv_used, 0);
        mpi::broadcast(world, recv_curr_sol, 0);
        mpi::broadcast(world, recv_best_sol, 0);
        double best_cost = std::numeric_limits<double>::infinity(); // Melhor custo atual

        backtrack(1, 0, recv_curr_sol, best_cost, recv_best_sol, recv_used, recv_points_distance, recv_N, id, n_machines);

        mpi::gather(world, best_cost, 0);
        mpi::gather(world, recv_best_sol, 0);
    }

    return 0;
}