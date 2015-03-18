// helper function
#ifndef CAFFE_UTIL_ANDREJ_UTIL_H_
#define CAFFE_UTIL_ANDREJ_UTIL_H_

using namespace std;

vector<string> strsplit_andrej(string str, string delim) { 
  int start = 0;
  int end; 
  vector<string> v; 
  while( (end = str.find(delim, start)) != string::npos )
  { 
        v.push_back(str.substr(start, end-start)); 
        start = end + delim.length(); 
  } 
  v.push_back(str.substr(start)); 
  return v; 
}

#endif
