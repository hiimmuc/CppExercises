/**
*    Author: Andrew
*    Created: 26.06.2024 19:09:02
**/

#include <bits/stdc++.h>

using namespace std;

#define FOR(i,a,b) for(int i = (a);i <= (b);++i)
#define FOD(i,b,a) for(int i = (b);i >= (a);--i)
#define REP(i,a,b) for(int i = (a) ;i < (b);++i)
#define ALL(x) ((x).first,(x).second)
#define fi first
#define se second

typedef long long ll;

typedef pair<int,int> pii;
typedef vector<pair<int,int>> vii;
typedef vector<vector<int>> vvi;

const int N = 1e6;

int n;
int a[N + 2],is_prime[N + 2],pre[N + 2],C[N + 2];
vector<int> luusnt;

void sieve(){
    fill(is_prime,is_prime + N,1);
    is_prime[0] = 0;
    is_prime[1] = 0;
    for (int i = 2 ; i* i <= N;i++){
        if (is_prime[i]){
            for(int j = i * i;j <= N;j+=i){
                is_prime[j] = 0;
            }
        }
    }
    FOR(i,1,n){
        if (is_prime[i]){
            luusnt.push_back(i);
        }
    }
}

// void sub1(){
//     int mx_sum = INT_MIN;
//     FOR(l,2,n - 1){
//         FOR(r,l + 1,n){
//             if (is_prime[l] && is_prime[r]){
//                 int sum = 0;
//                 FOR(k,l,r){
//                     sum += a[k];
//                 }
//                 mx_sum = max(mx_sum,sum);
//             }
//         }
//     }
//     cout << mx_sum << '\n';
// }

// void sub2(){
//     int mx_sum = INT_MIN;
//     FOR(i,1,n) pre[i] = pre[i - 1] + a[i];

//     FOR(i,0,(int)luusnt.size() - 1){
//         FOR(j,i + 1,(int)luusnt.size() - 1){
//             mx_sum = max(mx_sum,pre[luusnt[j]] - pre[luusnt[i] - 1]);
//         }
//     }
//     cout << mx_sum << "\n";
// }

void sub3(){
    int d = 0;
    FOR(i,1,n) pre[i] = pre[i - 1] + a[i];
    FOR(i,1,n){
        if (is_prime[i]){
            ++d;
            C[d] = i;
        }
    }
    int minn = 1e9,ans = -1e9;
    for(int i = 1; i <= d;i++){
        minn = min(minn,pre[C[i] - 1]);
        ans = max(ans,pre[C[i]] - minn);
    }
    cout << ans;
}

signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cin >> n;
    FOR(i,1,n) cin >> a[i];
    sieve();
    // sub1();
    // sub2();
    sub3();
    return 0;
}

/**Genius is 1% talent and 99% percent hard work**/
