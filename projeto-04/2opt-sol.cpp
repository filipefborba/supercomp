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

void opt_sol(std::vector<int> &solution, double &best_cost, std::vector<std::vector<double>> &points_distance, int N, int seed)
{
    std::vector<int> new_solution(N); // Nova solucao a ser criada
    double solution_cost = 0;         // Custo total dessa solucao

    // Preenche a solucao em ordem para que possamos permutar depois
    for (int a = 0; a < N; a++)
    {
        new_solution[a] = a;
    }

    int idx;
    // Realiza a permutacao da solucao
    for (int b = 1; b < N; b++)
    {
        idx = (int)((rand() % (((N - 1) - b) + 1)) + b);                        // Pegar um indice aleatorio entre 1 e N-1
        std::swap(new_solution[b], new_solution[idx]);                          // Swap dos elementos do vetor e salva no vetor de solucoes
        solution_cost += points_distance[new_solution[b - 1]][new_solution[b]]; // Calculo das distancias
    }
    solution_cost += points_distance[new_solution[0]][new_solution[N - 1]]; // Ultimo calculo: primeiro e ultimo

    // 2opt - Descruzar os segmentos
    double new_cost = 0;
    for (int c = 1; c < N; c++)
    {
        for (int d = c + 1; d < N; d++)
        {
            std::swap(new_solution[c], new_solution[d]); // Swap dos elementos do vetor e salva no vetor de solucoes
            new_cost = total_cost(new_solution, points_distance, N);
            if (new_cost < solution_cost)
            {
                solution_cost = new_cost;
            }
            else
            {
                std::swap(new_solution[d], new_solution[c]); // Swap dos elementos do vetor e salva no vetor de solucoes
            }
        }
    }

    if (solution_cost < best_cost)
    {
        solution = new_solution;
        best_cost = solution_cost;
    }
}

int main(int argc, char *argv[])
{
    mpi::environment env(argc, argv);
    mpi::communicator world;
    int id = world.rank();
    int n_machines = world.size();

    // Inicializar o random
    srand(id);

    long nSols = 10000; // 10.000

    if (id == 0)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Preparacao para receber os dados do arquivo
        int N;
        std::cin >> N;
        std::vector<std::pair<double, double>> points(N);

        double x, y;
        std::pair<double, double> point;
        for (int i = 0; i < N; i++)
        {
            std::cin >> x;
            std::cin >> y;
            point.first = x;
            point.second = y;
            points[i] = point;
        }

        // ---------------------------------------------------------------------
        // Preparacao para pre-calcular as distancias
        std::vector<std::vector<double>> points_distance(N, std::vector<double>(N, 0.0));

        calc_dist(points, points_distance);

        // ---------------------------------------------------------------------
        // Broadcast das informacoes para outras maquinas
        mpi::broadcast(world, N, 0);
        mpi::broadcast(world, points_distance, 0);
        // ---------------------------------------------------------------------
        // Preparacao sortear solucoes e calcular custos
        std::vector<int> solution(N);                               // Melhor solucao atual
        double best_cost = std::numeric_limits<double>::infinity(); // Melhor custo atual

        for (int i = 0; i < nSols; i++)
        {
            opt_sol(solution, best_cost, points_distance, N, (i * id) + i);
        }

        std::vector<std::vector<int>> best_solutions(n_machines, std::vector<int>(N, 0));
        std::vector<double> best_costs(n_machines);

        mpi::gather(world, best_cost, best_costs, 0);
        mpi::gather(world, solution, best_solutions, 0);

        auto iter = std::min_element(best_costs.begin(), best_costs.end());
        int position = std::distance(best_costs.begin(), iter);
        best_cost = *iter;
        solution = best_solutions[position];

        auto end_time = std::chrono::high_resolution_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

// ---------------------------------------------------------------------
// Print do tempo e do melhor caminho
#ifdef TIME
        std::cout << msecTotal << std::endl;
        std::cout << "milisegundo(s)." << std::endl;
#endif

        std::cout << std::fixed << std::setprecision(5);
        std::cout << best_cost;
        std::cout << " 0" << std::endl;

        for (int i = 0; i < N; i++)
        {
            std::cout << solution[i] << ' ';
        }
        std::cout << std::endl;
    }
    else
    {
        int recv_N;
        std::vector<std::vector<double>> recv_points_distance;
        mpi::broadcast(world, recv_N, 0);
        mpi::broadcast(world, recv_points_distance, 0);

        std::vector<int> solution(recv_N);                          // Melhor solucao atual
        double best_cost = std::numeric_limits<double>::infinity(); // Melhor custo atual

        for (int i = 0; i < nSols; i++)
        {
            opt_sol(solution, best_cost, recv_points_distance, recv_N, i);
        }

        mpi::gather(world, best_cost, 0);
        mpi::gather(world, solution, 0);
    }

    return 0;
}