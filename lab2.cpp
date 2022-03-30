#include <iostream>
#include <fstream>
#include <omp.h>


struct CrsMatrix
{
    int N;
    int NZ;
    double* elements;
    int* col;
    int* row;
    int size;
};

void test_1();
void test_2();
void test_3(CrsMatrix A, double* b);

CrsMatrix& readFile()
{
    CrsMatrix matrix;
    std::ifstream file("bcsstk38.mtx");
    int num_row, num_col, num_lines;

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');

    // Read number of rows and columns
    file >> num_row >> num_col >> num_lines;
    std::cout << num_col << "     " << num_row;

    matrix.N = num_col;
    matrix.NZ = num_lines*2;
    matrix.elements = new double[num_lines*2];
    matrix.size = num_lines * 2;
    matrix.col = new int[num_lines*2];
    matrix.row = new int[num_lines*2];


    // Create 2D array and fill with zeros

    // fill the matrix with data
    for (int l = 0; l < num_lines; l++)
    {
        double data;
        int row, col;
        file >> row >> col >> data;
        matrix.col[l] = col;
        matrix.row[l] = row;
        matrix.elements[l] = data;

        //так как симметричная
        matrix.col[l+ num_lines] = row;
        matrix.row[l + num_lines] = col;
        matrix.elements[l + num_lines] = data;
    }

    file.close();
    return matrix;
}

int main()
{/*
    std::ifstream file("bcsstm34.mtx");
    int num_row, num_col, num_lines;

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');

    // Read number of rows and columns
    file >> num_row >> num_col >> num_lines;
    std::cout << num_col << "     " << num_row;

    // Create 2D array and fill with zeros
    double* matrix;
    matrix = new double[num_row * num_col];
    std::fill(matrix, matrix + num_row * num_col, 0.);

        // fill the matrix with data
        for (int l = 0; l < num_lines; l++)
        {
            double data;
            int row, col;
            file >> row >> col >> data;
            matrix[(row - 1) + (col - 1) * num_row] = data;
        }

    file.close();
    */
    
    // Вычисляем сумму квадратов элементов вектора F
    CrsMatrix matrix = readFile();
    double* b = new double[matrix.N];
    for (size_t i = 0; i < matrix.N; i++)
    {
        b[i] = std::rand();
    }
    test_3(matrix, b);


    /*
    // Задаем начальное приближение корней
#pragma omp parallel for
    for (i = 0; i < size; i++) {
        x_k[i] = 0;
    }

#pragma omp parallel for
    for (i = 0; i < size; i++) {
        d_k[i] = 0;
    }

#pragma omp parallel for
    for (i = 0; i < size; i++) {
        g_k[i] = 0;
    }*/

    /*
    CrsMatrix matrix = readFile();

    double* x_k = new double[matrix.N];
    double* x_k_prev = new double[matrix.N];
    std::fill(x_k_prev, x_k_prev + matrix.N, 0.);


    //вычисление
    //пусть n циклов хватит

    for (size_t i = 0; i < matrix.N; i++)
    {
        std::fill(x_k, x_k + matrix.N, 0.);
        
    }


    /*
    for (int i = 0; i < num_row; i++)
    {
        for (int j = 0; j < num_col; j++)
            if (matrix[i + j * num_row] != 0)

                std::cout << i+1 << " " << j+1 << " " << matrix[i + j * num_row] << "\n";
    }
    */

}

//скалярное произведение тест  ~~1,85
void test_1()
{
    
    
    int size = 100000;
    double* a = new double[size];

    double* b = new double[size];

    std::fill(a, a + size, std::rand() + std::rand()/100.0);
    std::fill(b, b + size, std::rand() + std::rand()/100.0);
    
   
    omp_set_dynamic(0);
    omp_set_num_threads(8);

    
    double res = 0;
    double tmp;

    double count = 5000;

    //1000 норм кароч
    int chunk = 1000;
    
    double start = omp_get_wtime();
    for (size_t i = 0; i < count; i++)
    {
        for (int i = 0; i < size; i++) {
            tmp = a[i] * b[i];
            res += tmp;
        }
    }
    double kok = omp_get_wtime() - start;
    
    start = omp_get_wtime();
    for (size_t i = 0; i < count; i++)
    {
#pragma omp parallel shared(res) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) reduction(+:res) nowait
            for (int i = 0; i < size; i++) {
                tmp = a[i] * b[i];
                res += tmp;
            }
        }
    }
    std::cout << kok / (omp_get_wtime() - start) << "\n";
}

//заполнение тест говно кароч филлом лучше
void test_2()
{
    omp_set_dynamic(0);
    omp_set_num_threads(2);
    
    int size = 100000;
    double* a = new double[size];

    double* b = new double[size];
    double count = 5000;

    double start = omp_get_wtime();
    for (size_t i = 0; i < count; i++)
    {
        std::fill(a, a + size, 0.0);
        std::fill(b, b + size, 0.0);

    }
   
    std::cout << omp_get_wtime() - start << "\n";

    start = omp_get_wtime();
    int chunk = 1000;
    for (size_t i = 0; i < count; i++)
    {
#pragma omp parallel
        {
#pragma omp for schedule(dynamic, chunk) nowait
        for (int i = 0; i < size; i++)
        {
            a[i] = 0.0;
        }
        };
    }
    std::cout << omp_get_wtime() - start << "\n";


}

//test градиента
void test_3(CrsMatrix A, double* b)
{
    omp_set_dynamic(0);
    omp_set_num_threads(2);

    int size = A.size;
    double* x_k_prev = new double[A.N];
    std::fill(x_k_prev, x_k_prev + A.N, 0.);

    double count = 1;
    double res = 0;
    double* res1 = new double[A.N];
    double* res2 = new double[A.N];
    double tmp;
    //50 норм кароч
    int chunk = 50;

    double start = omp_get_wtime();
    std::cout << "считаю\n";
    for (size_t k = 0; k < count; k++)
    {
        for (int j = 0; j < size; j++) {
            auto el = A.elements[j];
            tmp = 0;
            for (int i = 0; i < A.N; i++)
            {
                tmp += el * b[i];
            }
            res1[A.elements] = tmp;
        }
    }
    double kok = omp_get_wtime() - start;
    std::cout << res << "\n";
    res = 0;

    start = omp_get_wtime();
    for (int k = 0; k < count; k++)
    {
#pragma omp parallel shared(res) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) reduction(+:res) nowait
            for (int i = 0; i < size; i++) {
                auto el = A.elements[i];
                tmp = 0;
                for (int j = 0; j < A.N; j++)
                {
                    tmp += el * b[j];
                }
                res2[i] = tmp;
            }
        }
    }
    std::cout << "asdasda";
    std::cout << kok / (omp_get_wtime() - start) << "\n" << res;
}



void grad(CrsMatrix A, double* b, int* jptr, int* iptr, int size, int pot, double toch, int size1)
{
    double start = omp_get_wtime();
    double* x_k = new double[size];
    double* x_k_prev = new double[size];

    double* d_k = new double[size];
    double* d_k_prev = new double[size];

    double* g_k = new double[size];
    double* g_k_prev = new double[size];

    float* Zk = new float[size];
    float* Rk = new float[size];
    float* Sz = new float[size];
    float alpha, beta, mf;
    float Spr, Spr1, Spz;
    int i, j, kl = 1;
    double max_iter = size;

    //Начальное приближение
    std::fill(x_k_prev, x_k_prev + size, 0.);
    std::fill(d_k_prev, d_k_prev + size, 0.);
    for (size_t i = 0; i < size; i++)
        g_k_prev[i] = -b[i];


    omp_set_num_threads(1);

    
    /*

    int Iteration = 0;
    do {
        Iteration++;
        // Вычисляем числитель и знаменатель для коэффициента
        // alpha = (rk-1,rk-1)/(Azk-1,zk-1) 
        Spz = 0;
        Spr = 0;


        for (i = 0; i < size1 - 1; i++)
        {
            Sz[i] = 0;
            for (j = iptr[i]; j < iptr[i + 1]; j++)
                Sz[i] += Zk[jptr[j]] * aelem[j];
        }



        for (i = 0; i < size1 - 1; i++) {

            Spz += Sz[i] * Zk[i];
            Spr += Rk[i] * Rk[i];
        }
        alpha = Spr / Spz;          //    alpha    


                                    // Вычисляем вектор решения: xk = xk-1+ alpha * zk-1,
                                    //вектор невязки: rk = rk-1 - alpha * A * zk-1 и числитель для betaa равный (rk,rk) 
        Spr1 = 0;

        for (i = 0; i < size1 - 1; i++) {
            Xk[i] += alpha * Zk[i];
            Rk[i] -= alpha * Sz[i];
            Spr1 += Rk[i] * Rk[i];

        }

        kl++;


        // Вычисляем  beta  
        beta = Spr1 / Spr;


        // Вычисляем вектор спуска: zk = rk+ beta * zk-1 

        for (i = 0; i < size1 - 1; i++)
            Zk[i] = Rk[i] + beta * Zk[i];
    }
    // Проверяем условие выхода из итерационного цикла  
    while (Spr1 / mf > toch * toch && Iteration < max_iter);
    ofstream file1("resh1000.txt");
#pragma omp critical 
    {
        /*cout << "Вектор - решение" << endl;
        for (int i = 0; i < size1-1; i++)
        {
            cout << Xk[i] << endl;
            file1 << Xk[i] << endl;
        }*//*
    }
    file1.close();
    delete[] Xk, Rk, Sz, Zk;*/
}
