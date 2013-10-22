#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

/** Class for computing the mean and std_err of a sequence incrementally */
class inc_stat {
  int n;
  double _mean, _M2;
public:
  inc_stat() : n(0), _mean(0), _M2(0) {}
  void update(double x) {
    ++n;
    const double delta = x - _mean;
    _mean += delta/n;
    _M2 += delta*(x - _mean);
  }
  auto mean() const -> double { return _mean; }
  auto variance() const -> double { return _M2/(n-1); }
  auto std_dev() const -> double { return std::sqrt(variance()); }
  auto std_err() const -> double { return std_dev() / std::sqrt(n); }
  auto total() const -> double { return n*mean(); }
};

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Wrong number of arguments\n";
    return 1;
  }
  std::string base_name(argv[1]);
  int num_episodes = atoi(argv[2]);
  int head_skip = atoi(argv[3]);
  
  std::vector<inc_stat> return_statistics(num_episodes);
  std::vector<inc_stat> cumulative_statistics(num_episodes);

  int num_failures = 0;
  int max_failures = 30;
  int file_number = 0;

  while (num_failures < max_failures) {
    //std::cerr << "Loading: " << (base_name + "-" + std::to_string(file_number) + ".txt\n");
    std::ifstream in(base_name + "-" + std::to_string(file_number) + ".txt");
    if (!in.is_open()) { ++num_failures; }
    else {
      for (int i = 0; i < head_skip; ++i) {
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
      double _return = 0;
      double total_return = 0;
      for (int i = 0; in >> _return; ++i) {
        return_statistics[i].update(_return);
        total_return += _return;
        cumulative_statistics[i].update(total_return);
      }
    }
    ++file_number;
  }
  
  std::cout << "# " << base_name << " (" << (file_number - max_failures) << " runs)\n";
  for (int i = 0; i < num_episodes; ++i) {
    std::cout << return_statistics[i].mean() 
         << " " << return_statistics[i].std_err()
         << " " << cumulative_statistics[i].std_err()
         << "\n";
  }
}
