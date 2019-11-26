#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <vector>
#include <utility>

double dist(std::pair<double, double> &p1, std::pair<double, double> &p2)
{
    return sqrt(pow((p1.first - p2.first), 2) + pow((p1.second - p2.second), 2));
}

double path_dist(std::vector<int> &seq, std::vector<std::pair<double, double>> &points)
{
    double d = dist(points[seq.back()], points[seq[0]]);
    for (int i = 0; i < seq.size() - 1; i++)
    {
        d += dist(points[seq[i]], points[seq[i + 1]]);
    }
    return d;
}

double backtrack(std::vector<std::pair<double, double>> &points, int idx, double curr_cost, std::vector<int> &curr_sol, double best_cost, std::vector<int> &best_seq, std::vector<bool> &used)
{
    if (idx == points.size())
    {
        curr_cost += dist(points[curr_sol[0]], points[curr_sol.back()]);
        if (curr_cost < best_cost)
        {
            best_seq = curr_sol;
            best_cost = curr_cost;
        }
        return best_cost;
    }

    for (int i = 0; i < points.size(); i++)
    {
        if (used[i] == false)
        {
            used[i] = true;
            curr_sol[idx] = i;

            double new_cost = curr_cost + dist(points[curr_sol[idx - 1]], points[curr_sol[idx]]);
            best_cost = backtrack(points, idx + 1, new_cost, curr_sol, best_cost, best_seq, used);

            used[i] = false;
            curr_sol[idx] = -1;
        }
    }
    return best_cost;
}

int main()
{
    int N;
    std::cin >> N;
    std::vector<std::pair<double, double>> points(N);
    std::vector<bool> used(N);
    std::vector<int> curr_sol(N);
    std::vector<int> best_sol(N);

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

    auto start_time = std::chrono::high_resolution_clock::now();
    backtrack(points, 1, 0, curr_sol, std::numeric_limits<double>::infinity(), best_sol, used);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

#ifdef TIME
    std::cout << runtime.count() << std::endl;
    std::cout << "milisegundo(s)." << std::endl;
#endif

    std::cout << std::fixed << std::setprecision(5);
    std::cout << path_dist(best_sol, points);
    std::cout << " 1" << std::endl;

    for (int i = 0; i < best_sol.size(); i++)
    {
        std::cout << best_sol[i] << ' ';
    }
    std::cout << std::endl;

    return 0;
}