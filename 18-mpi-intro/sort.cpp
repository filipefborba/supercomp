#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <iterator>
#include <algorithm>
#include <random>
#include <functional>
namespace mpi = boost::mpi;

int generate_random()
{
    return std::rand() % 100;
}

int main(int argc, char *argv[])
{
    mpi::environment env(argc, argv);
    mpi::communicator world;
    int i = world.rank();
    int N = world.size();

    if (i == 0)
    {
        std::srand(0);

        int vec_size = 100;
        std::vector<int> v_random(vec_size);
        std::generate(v_random.begin(), v_random.end(), generate_random);

        int chunk_size = ceil(vec_size / N);

        for (int j = 1; j < N; j++)
        {
            std::vector<int> v_slice(v_random.begin() + (j * chunk_size), v_random.begin() + ((1 + j) * chunk_size));
            world.send(j, 0, v_slice);
        }

        std::vector<int> v_initial(v_random.begin(), v_random.begin() + chunk_size);
        std::sort(v_initial.begin(), v_initial.end());

        std::vector<int> v_dest;
        std::vector<int> v_sorted;
        for (int j = 1; j < N; j++)
        {
            std::vector<int> v_recv_sorted;
            world.recv(j, 0, v_recv_sorted);

            if (j == 1)
            {
                std::merge(v_recv_sorted.begin(), v_recv_sorted.end(), v_initial.begin(), v_initial.end(), std::back_inserter(v_sorted));
                v_sorted = v_dest;
            }
            else
            {
                v_dest.clear();
                std::merge(v_recv_sorted.begin(), v_recv_sorted.end(), v_sorted.begin(), v_sorted.end(), std::back_inserter(v_dest));
                v_sorted = v_dest;
            }
        }

        for (int k = 0; k < vec_size; k++)
        {
            std::cout << v_sorted[k] << ' ';
        }
        std::cout << std::endl;
    }
    else
    {
        std::vector<int> v_recv;
        world.recv(0, 0, v_recv);

        std::sort(v_recv.begin(), v_recv.end());
        world.send(0, 0, v_recv);
    }

    return 0;
}