#include <vector>
#include <utility>
#include <iostream>
#include <cstring>


extern "C" {
  struct VecPair{
    int* vec1;
    std::size_t vec1_size;
    int* vec2;
    std::size_t vec2_size;
  };


  VecPair GetIndexes(const int* idxs, std::size_t idxs_size, int n, int k) {
    std::vector<int> rows;
    std::vector<int> cols;

    std::vector<std::vector<int>> v(n, std::vector<int>());
    for (int i = 0; i < idxs_size; ++i) {
      v[idxs[i]].push_back(i);
    }
    /*
    for (int i = 0; i < n; ++i) {
      std::vector<int> res;
      for (int j = 0; j < idxs_size; ++j) {
        if (idxs[j] == i) {
          res.push_back(j);
        }
      }
      std::vector<int> local_cols(res.size());
      for (int j = 0; j < res.size(); ++j) {
        local_cols[j] = k*(res[j] / k) + (res[j] % k);
      }
      std::vector<int> tmp(res.size(), i);
      rows.insert(rows.end(), tmp.begin(), tmp.end());
      cols.insert(cols.end(), local_cols.begin(), local_cols.end());
    }
    */

    int* rows_buf = new int[idxs_size];
    int* cols_buf = new int[idxs_size];
    int offset = 0;
    for (int i = 0; i < n; ++i) {
      int l = v[i].size();
      std::fill(rows_buf+offset, rows_buf+offset+l, i);
      std::memcpy(cols_buf+offset, v[i].data(), l * sizeof(int));
      offset += l; 
    }
    
    VecPair vec_pair{};
    vec_pair.vec1 = rows_buf;
    vec_pair.vec1_size = idxs_size;
    vec_pair.vec2 = cols_buf;
    vec_pair.vec2_size = idxs_size;

    return vec_pair;
  }


    // Function to free the allocated memory
    void freeVecPair(VecPair vp) {
      delete[] vp.vec1;
      delete[] vp.vec2;
    }
}
