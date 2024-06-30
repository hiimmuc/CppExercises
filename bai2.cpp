
#include "stdc++.h"

#define ll long long
#define ull unsigned long long
#define fast ios_base::sync_with_stdio(0) ; cin.tie(0) ; cout.tie(0) ;
#define fi first
#define se second
#define bit(mask , i) ((mask >> i) & 1)
using namespace std;
const ll N = 1e6 + 5 , mod = 10 , inf = LLONG_MAX ;
struct matrix
{
    vector<vector<ll>> mat;
    int col = 0 , row = 0;
    matrix(int n , int m){
        col = n; row = m;
        mat.resize(col , vector<ll>(row , 0));
    }
    matrix operator * (const matrix &b) const
    {
        matrix a = *this;
        matrix c(a.col , b.row);
        int sz = a.row ; // sz = a.row = b.col ;
        for(int i = 0 ; i < a.col ; i++)
            for(int j = 0 ; j < b.row ; j++)
                for(int k = 0 ; k < sz ; k++)
            {
                (c.mat[i][j] += a.mat[i][k] * b.mat[k][j]) %= mod;
            }
        return c;
    }
    matrix power(ll n) const //only ma tran vuong
    {
        matrix a = *this;
        matrix ans(a.col , a.col);
        for(int i = 0 ; i < a.col ; i++)
            ans.mat[i][i] = 1;
        while(n)
        {
            if(n & 1) ans = ans * a;
            a = a * a;
            n /= 2;
        }
        return ans;
    }
};
ll get(ll a , ll b , ll n)
{
    if(n == 0)return 0;
    if(n == 1)return a % mod;
    if(n == 2)return (a + b) % mod;
    matrix f(3 , 1);
    f.mat = {
        {(a + b) % mod},
        {b % mod},
        {a % mod}
    };
    matrix T(3 , 3);
    T.mat ={
        {1 , 1 , 1},
        {0 , 1 , 1},
        {0 , 1 , 0}
    };
    matrix ans = T.power(n - 2) * f;
    return ans.mat[0][0];
}
void sol()
{
    ll a , b , l , r;
    cin >> a >> b >> l >> r;
    int u = get(a , b , l - 1);
    int v = get(a , b , r);
    cout << (v - u + mod) % mod;
}
int main()
{
    fast
    ll t; cin >> t;
    while(t--)
    {
        sol();
        cout << '\n';
    }
    return 0;
}
