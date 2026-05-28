#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <complex>
#include <string>
#include <cstddef>

using namespace std;
typedef size_t       sz_t;
typedef vector<vector<double>> Mat;
typedef vector<double>         Vec;
typedef vector<complex<double>> CVec;

const double EPS = 1e-9;

void printMatrix(const Mat& M, const string& name = "") {
    if (!name.empty()) cout << name << " =\n";
    sz_t n = M.size();
    for (sz_t i = 0; i < n; i++) {
        cout << "  [ ";
        for (sz_t j = 0; j < n; j++)
            cout << setw(10) << fixed << setprecision(4) << M[i][j] << " ";
        cout << "]\n";
    }
}

Mat matMul(const Mat& A, const Mat& B) {
    sz_t n = A.size();
    Mat C(n, Vec(n, 0));
    for (sz_t i = 0; i < n; i++)
        for (sz_t j = 0; j < n; j++)
            for (sz_t k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

double niceRound(double v) { return fabs(v) < EPS ? 0.0 : v; }

// Format a number cleanly: show as integer if it is one, else 2 decimal places
string fmtNum(double v) {
    v = niceRound(v);
    if (fabs(v - round(v)) < 1e-6)
        return to_string(static_cast<int>(round(v)));
    // 2 decimal places
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f", v);
    return string(buf);
}

//  Step 0 – Input
Mat inputMatrix(int& n) {
    cout << "Enter matrix size (2 for 2x2, 3 for 3x3): ";
    while (!(cin >> n) || (n != 2 && n != 3)) {
        cin.clear(); cin.ignore(1000, '\n');
        cout << "Invalid. Enter 2 or 3: ";
    }
    sz_t N = static_cast<sz_t>(n);
    Mat A(N, Vec(N));
    cout << "Enter elements of matrix A row by row (" << n << "x" << n << "):\n";
    for (sz_t i = 0; i < N; i++) {
        for (sz_t j = 0; j < N; j++)
            cin >> A[i][j];
    }
    return A;
}


//  Step 1 – Eigenvalues
CVec eigenvalues2x2(const Mat& A) {
    double tr   = A[0][0] + A[1][1];
    double det  = A[0][0]*A[1][1] - A[0][1]*A[1][0];
    double disc = tr*tr - 4*det;
    CVec ev;
    if (disc >= 0) {
        ev.push_back({(tr + sqrt(disc))/2, 0});
        ev.push_back({(tr - sqrt(disc))/2, 0});
    } else {
        ev.push_back({tr/2,  sqrt(-disc)/2});
        ev.push_back({tr/2, -sqrt(-disc)/2});
    }
    return ev;
}

CVec eigenvalues3x3(const Mat& A) {
    double a=A[0][0], b=A[0][1], c=A[0][2];
    double d=A[1][0], e=A[1][1], f=A[1][2];
    double g=A[2][0], h=A[2][1], k=A[2][2];

    double p  = -(a+e+k);
    double q  = (a*e - b*d) + (a*k - c*g) + (e*k - f*h);
    double r  = -(a*(e*k - f*h) - b*(d*k - f*g) + c*(d*h - e*g));
    double p2 = p/3.0;
    double qq = q - p*p/3.0;
    double rr = r - p*q/3.0 + 2*p*p*p/27.0;
    double D  = qq*qq*qq/27.0 + rr*rr/4.0;

    CVec roots;
    const double PI = acos(-1.0);

    if (fabs(D) < EPS) {
        double u = (rr >= 0) ? -cbrt(rr/2) : cbrt(-rr/2);
        roots.push_back({2*u - p2, 0});
        roots.push_back({  -u - p2, 0});
        roots.push_back({  -u - p2, 0});
    } else if (D > 0) {
        double sqD = sqrt(D);
        double A3  = cbrt(-rr/2 + sqD);
        double B3  = cbrt(-rr/2 - sqD);
        roots.push_back({A3+B3-p2, 0});
        roots.push_back({-(A3+B3)/2-p2,  (A3-B3)*sqrt(3.0)/2});
        roots.push_back({-(A3+B3)/2-p2, -(A3-B3)*sqrt(3.0)/2});
    } else {
        double m     = 2*sqrt(-qq/3.0);
        double theta = acos(3*rr/(qq*m)) / 3.0;
        roots.push_back({m*cos(theta)           - p2, 0});
        roots.push_back({m*cos(theta + 2*PI/3)  - p2, 0});
        roots.push_back({m*cos(theta + 4*PI/3)  - p2, 0});
    }
    return roots;
}

void printEigenvalues(const CVec& ev) {
    cout << "\nStep 1: Eigenvalues\n";
    for (sz_t i = 0; i < ev.size(); i++) {
        cout << "  Eigenvalue (λ" << i+1 << ") = ";
        double re = niceRound(ev[i].real());
        double im = niceRound(ev[i].imag());
        if (fabs(im) < EPS)
            cout << re << "\n";
        else
            cout << re << (im>=0?"+":"") << im << "i\n";
    }
}

void rref(Mat& M, sz_t n) {
    sz_t pivot_row = 0;
    for (sz_t col = 0; col < n && pivot_row < n; col++) {
        sz_t piv = n; // sentinel
        for (sz_t row = pivot_row; row < n; row++)
            if (fabs(M[row][col]) > EPS) { piv = row; break; }
        if (piv == n) continue;
        swap(M[pivot_row], M[piv]);
        double div = M[pivot_row][col];
        for (sz_t j = 0; j <= n; j++) M[pivot_row][j] /= div;
        for (sz_t row = 0; row < n; row++) {
            if (row == pivot_row) continue;
            double factor = M[row][col];
            for (sz_t j = 0; j <= n; j++)
                M[row][j] -= factor * M[pivot_row][j];
        }
        pivot_row++;
    }
}

//  Step 2 – Eigenvectors
//  Returns ALL basis vectors of null(A - λI)
// Returns a list of basis eigenvectors for eigenvalue lambda
// (one vector per free variable in RREF of (A - λI))
Mat eigenvectorBasis(const Mat& A, double lambda) {
    sz_t n = A.size();
    Mat M(n, Vec(n+1, 0));
    for (sz_t i = 0; i < n; i++)
        for (sz_t j = 0; j < n; j++)
            M[i][j] = A[i][j] - (i==j ? lambda : 0.0);
    rref(M, n);

    // Find pivot columns
    vector<sz_t> pivot_col_of_row(n, n); // n = "no pivot"
    for (sz_t i = 0; i < n; i++)
        for (sz_t j = 0; j < n; j++)
            if (fabs(M[i][j] - 1.0) < EPS) { pivot_col_of_row[i] = j; break; }

    vector<bool> is_pivot(n, false);
    for (sz_t i = 0; i < n; i++)
        if (pivot_col_of_row[i] != n) is_pivot[pivot_col_of_row[i]] = true;

    vector<sz_t> free_cols;
    for (sz_t j = 0; j < n; j++)
        if (!is_pivot[j]) free_cols.push_back(j);

    Mat basis;
    // For each free variable, set it to 1 and others to 0, back-substitute
    for (sz_t fi = 0; fi < free_cols.size(); fi++) {
        Vec vec(n, 0);
        // Set all free vars to 0, except the current one = 1
        for (sz_t fj = 0; fj < free_cols.size(); fj++)
            vec[free_cols[fj]] = (fi == fj) ? 1.0 : 0.0;
        // Back-substitute for pivot variables
        for (int i = static_cast<int>(n)-1; i >= 0; i--) {
            sz_t si = static_cast<sz_t>(i);
            if (pivot_col_of_row[si] == n) continue; // no pivot in this row
            sz_t pc = pivot_col_of_row[si];
            double val = 0;
            for (sz_t j = 0; j < n; j++)
                if (j != pc) val -= M[si][j] * vec[j];
            vec[pc] = niceRound(val);
        }
        basis.push_back(vec);
    }
    return basis;
}

void printEigenvectors(const Mat& A, const CVec& ev, Mat& allEigvecs) {
    cout << "\nStep 2: Eigenvectors\n";
    sz_t n = A.size();
    allEigvecs.clear();

    // Track which eigenvalues we've already processed (avoid duplicates for repeated λ)
    vector<double> processed;

    for (sz_t i = 0; i < ev.size(); i++) {
        double re = ev[i].real();
        double im = niceRound(ev[i].imag());

        if (fabs(im) > EPS) {
            cout << "  λ" << i+1 << " = " << niceRound(re)
                 << (im>=0?"+":"") << im << "i  →  complex eigenvalue (not real).\n";
            continue;
        }

        // Skip if we already processed this eigenvalue
        bool already = false;
        for (double d : processed) if (fabs(d-re) < EPS) { already=true; break; }
        if (already) continue;
        processed.push_back(re);

        Mat basis = eigenvectorBasis(A, re);
        if (basis.empty()) {
            cout << "  λ = " << niceRound(re) << "  →  Only trivial solution.\n";
            continue;
        }

        for (sz_t bi = 0; bi < basis.size(); bi++) {
            // Format: {t(x,y,...) : t≠0}
            string vstr = "{t(";
            for (sz_t k = 0; k < n; k++) {
                vstr += fmtNum(basis[bi][k]);
                if (k < n-1) vstr += ",";
            }
            vstr += ") : t≠0}";
            cout << "  " << vstr << " is the set of eigenvectors of A "
                 << "corresponding to the eigenvalue λ = " << niceRound(re) << "\n";
            allEigvecs.push_back(basis[bi]);
        }
    }
    (void)n; // suppress unused warning
}

//  Step 3 – Diagonalizability
int geomMultiplicity(const Mat& A, double lambda) {
    sz_t n = A.size();
    Mat M(n, Vec(n+1, 0));
    for (sz_t i = 0; i < n; i++)
        for (sz_t j = 0; j < n; j++)
            M[i][j] = A[i][j] - (i==j ? lambda : 0.0);
    rref(M, n);
    int rank = 0;
    for (sz_t i = 0; i < n; i++)
        for (sz_t j = 0; j < n; j++)
            if (fabs(M[i][j]) > EPS) { rank++; break; }
    return static_cast<int>(n) - rank;
}

bool isDiagonalizable(const Mat& A, const CVec& ev, string& reason) {
    int n = static_cast<int>(A.size());

    for (sz_t i = 0; i < ev.size(); i++)
        if (fabs(niceRound(ev[i].imag())) > EPS) {
            reason = "Matrix has complex (non-real) eigenvalues, "
                     "so it is NOT diagonalizable over ℝ.";
            return false;
        }

    Vec distinct;
    for (sz_t i = 0; i < ev.size(); i++) {
        double re = ev[i].real();
        bool found = false;
        for (double d : distinct) if (fabs(d-re) < EPS) { found=true; break; }
        if (!found) distinct.push_back(re);
    }

    int totalGeo = 0;
    for (double lam : distinct) {
        int alg = 0;
        for (sz_t i = 0; i < ev.size(); i++)
            if (fabs(ev[i].real()-lam) < EPS) alg++;
        int geo = geomMultiplicity(A, lam);
        totalGeo += geo;
        if (geo < alg) {
            reason = "For λ = " + fmtNum(lam) +
                     ": algebraic multiplicity = " + to_string(alg) +
                     ", but geometric multiplicity = " + to_string(geo) +
                     ". Since geo < alg, the matrix is NOT diagonalizable.";
            return false;
        }
    }
    if (totalGeo < n) {
        reason = "Total independent eigenvectors (" + to_string(totalGeo) +
                 ") < n (" + to_string(n) + "). Matrix is NOT diagonalizable.";
        return false;
    }
    reason = "Each eigenvalue has geometric = algebraic multiplicity, "
             "and there are " + to_string(n) +
             " linearly independent eigenvectors. Matrix IS diagonalizable.";
    return true;
}

void printDiagCheck(bool diag, const string& reason) {
    cout << "\nStep 3: Diagonalizability Check \n";
    cout << "  " << reason << "\n";
    cout << "  → Result: A is " << (diag ? "" : "NOT ") << "diagonalizable.\n";
}

//  Step 4 – Matrix P
Mat buildP(const Mat& eigvecs, sz_t n) {
    Mat P(n, Vec(n, 0));
    for (sz_t j = 0; j < eigvecs.size() && j < n; j++) {
        if (eigvecs[j].empty()) continue;
        for (sz_t i = 0; i < n; i++)
            P[i][j] = eigvecs[j][i];
    }
    return P;
}

//  Step 5 – Inverse of P (Gauss-Jordan)
bool invertMatrix(const Mat& P, Mat& Pinv) {
    sz_t n = P.size();
    Mat aug(n, Vec(2*n, 0));
    for (sz_t i = 0; i < n; i++) {
        for (sz_t j = 0; j < n; j++) aug[i][j] = P[i][j];
        aug[i][n+i] = 1.0;
    }
    for (sz_t col = 0; col < n; col++) {
        sz_t piv = n;
        for (sz_t row = col; row < n; row++)
            if (fabs(aug[row][col]) > EPS) { piv = row; break; }
        if (piv == n) return false;
        swap(aug[col], aug[piv]);
        double d = aug[col][col];
        for (sz_t j = 0; j < 2*n; j++) aug[col][j] /= d;
        for (sz_t row = 0; row < n; row++) {
            if (row == col) continue;
            double f = aug[row][col];
            for (sz_t j = 0; j < 2*n; j++)
                aug[row][j] -= f * aug[col][j];
        }
    }
    Pinv.assign(n, Vec(n));
    for (sz_t i = 0; i < n; i++)
        for (sz_t j = 0; j < n; j++)
            Pinv[i][j] = niceRound(aug[i][n+j]);
    return true;
}

//  Step 6 – Verify D = P⁻¹ A P
bool isNearDiagonal(const Mat& M) {
    sz_t n = M.size();
    for (sz_t i = 0; i < n; i++)
        for (sz_t j = 0; j < n; j++)
            if (i != j && fabs(M[i][j]) > 1e-6) return false;
    return true;
}

void verifyDiag(const Mat& A, const Mat& P, const Mat& Pinv) {
    cout << "\nStep 6: Verify D = P⁻¹ × A × P\n";
    Mat D = matMul(Pinv, matMul(A, P));
    cout << "  Computed D = P⁻¹AP:\n";
    printMatrix(D);
    if (isNearDiagonal(D)) {
        cout << "\n  ✓ D is a diagonal matrix.\n";
        cout << "  ∴ A is diagonalizable and D = P⁻¹AP is confirmed.\n";
    } else {
        cout << "\n  ✗ Result is NOT diagonal – please check P.\n";
    }
}

int main() {
    int n;
    Mat A = inputMatrix(n);
    sz_t N = static_cast<sz_t>(n);

    cout << "\nMatrix A:\n";
    printMatrix(A, "A");

    // 1. Eigenvalues
    CVec ev = (n == 2) ? eigenvalues2x2(A) : eigenvalues3x3(A);
    printEigenvalues(ev);

    // 2. Eigenvectors
    Mat eigvecs;
    printEigenvectors(A, ev, eigvecs);

    // 3. Diagonalizability
    string reason;
    bool diag = isDiagonalizable(A, ev, reason);
    printDiagCheck(diag, reason);

    if (!diag) {
        // Reason already printed in Step 3 — no repeat
        return 0;
    }

    // 4. Matrix P
    Mat P = buildP(eigvecs, N);
    cout << "\nStep 4: Matrix P (columns = eigenvectors)\n";
    printMatrix(P, "P");

    // 5. Inverse of P
    Mat Pinv;
    cout << "\nStep 5: Inverse of P\n";
    if (!invertMatrix(P, Pinv)) {
        cout << "  P is singular (det = 0) → P⁻¹ does not exist.\n";
        cout << "  Reason: Eigenvectors are linearly dependent.\n";
        return 0;
    }
    printMatrix(Pinv, "P⁻¹");

    // 6. Verify D = P⁻¹AP
    verifyDiag(A, P, Pinv);
    return 0;
}