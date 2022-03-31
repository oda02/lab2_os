#include <iostream>
#include <fstream>
#include <omp.h>


struct MyMatrix
{
    int N;
    int size;

    double* elements;

    MyMatrix(int n): N(n) 
    {
        elements = new double[n * n];
        std::fill(elements, elements + n * n, 0.);
    }

    double* operator[](int i) {
        return &elements[i* N];
    }

};



void test_1();
void test_2();
void test_3(MyMatrix A, double* b);
void grad(MyMatrix A, double* b);

MyMatrix& readFile()
{
    
    std::ifstream file("bcsstk38.mtx");
    int num_row, num_col, num_lines;

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');

    // Read number of rows and columns
    file >> num_row >> num_col >> num_lines;
    std::cout << num_col << "     " << num_row;

    MyMatrix matrix(num_col);
    matrix.size = num_lines * 2;


    // Create 2D array and fill with zeros

    // fill the matrix with data
    for (int l = 0; l < num_lines; l++)
    {
        double data;
        int row, col;
        file >> row >> col >> data;
        matrix[row-1][col-1] = data;

        //так как симметричная
        matrix[col-1][row-1] = data;
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
    /*
    CrsMatrix matrix = readFile();
    double* b = new double[matrix.N];
    for (size_t i = 0; i < matrix.N; i++)
    {
        b[i] = std::rand();
    }
    test_3(matrix, b);*/


    /*MyMatrix matrix = readFile();
    double* b = new double[matrix.N];
    for (size_t i = 0; i < matrix.N; i++)
    {
        b[i] = std::rand();
    }*/
    //test_3(matrix, b); 


    MyMatrix matrix(3);
    matrix[0][0] = 2;
    matrix[0][1] = 1;
    matrix[0][2] = 1;
    matrix[1][0] = 1;
    matrix[1][1] = -1;
    matrix[1][2] = 0;
    matrix[2][0] = 3;
    matrix[2][1] = -1;
    matrix[2][2] = 2;

    double* b = new double[matrix.N];
    b[0] = 2;
    b[1] = -2;
    b[2] = 2;
    grad(matrix, b);

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
    
    
    int size = 8200;
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

//test градиента  ~~1.95

void test_3(MyMatrix A, double* b)
{
    omp_set_dynamic(0);
    omp_set_num_threads(1);

    double* x_k_prev = new double[A.N];
    std::fill(x_k_prev, x_k_prev + A.N, 0.);

    double count = 10;
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
        for (int i = 0; i < A.N; i++)
        {
            tmp = 0;
            for (int j = 0; j < A.N; j++) {
                auto el = A[i][j];

                tmp += el * x_k_prev[j];


            }
            res1[i] = tmp - b[i];
        }
    }
    double kok = omp_get_wtime() - start;
    //std::cout << kok << "\n";
    res = 0;

    start = omp_get_wtime();
    for (int k = 0; k < count; k++)
    {
#pragma omp parallel shared(res) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < A.N; i++)
            {
                tmp = 0;
                for (int j = 0; j < A.N; j++) {
                    auto el = A[i][j];

                    tmp += el * x_k_prev[j];


                }
                res2[i] = tmp - b[i];
            }
        }
    }
    std::cout << kok / (omp_get_wtime() - start) << "\n" << res;

    for (size_t i = 0; i < A.N; i++)
    {
        if (res1[i] != res2[i])
            std::cout << "hueta";
    }
}



void grad(MyMatrix A, double* b)
{
    int size = A.N;
    double start = omp_get_wtime();
    double* x_k = new double[size];
    //double* x_k_prev = new double[size];

    double* s_k = new double[size];
    //double* s_k_prev = new double[size];

    double* f_k = new double[size];
    //double* f_k_prev = new double[size];

    double* tmp_vec = new double[size];

    double tmp, scalar_p1, scalar_p2, beta_znam;
    int chunk;


    //Начальное приближение
    std::fill(x_k, x_k + size, 0.);
    for (size_t i = 0; i < size; i++)
    {
        s_k[i] = b[i];
        f_k[i] = -b[i];
    }

    //вычислим x_1
    scalar_p1 = 0;
    scalar_p2 = 0;

    //Знаменатель Матрица на вектор
    chunk = 50;
#pragma omp parallel shared(s_k) private(tmp)
    {
#pragma omp for schedule(dynamic, chunk) nowait
        for (int i = 0; i < A.N; i++)
        {
            tmp = 0;
            for (int j = 0; j < A.N; j++) {
                auto el = A[i][j];

                tmp += el * s_k[j];


            }
            tmp_vec[i] = tmp;
        }
    }

    // Знаменатель скалярное
    chunk = 1000;
#pragma omp parallel shared(scalar_p2) private(tmp)
    {
#pragma omp for schedule(dynamic, chunk) reduction(+:scalar_p2) nowait
        for (int i = 0; i < size; i++) {
            tmp = s_k[i] * tmp_vec[i];
            scalar_p2 += tmp;
        }
    }


    //Числитель

#pragma omp parallel shared(scalar_p1) private(tmp)
    {
#pragma omp for schedule(dynamic, chunk) reduction(+:scalar_p1) nowait
        for (int i = 0; i < size; i++) {
            tmp = f_k[i] * s_k[i];
            scalar_p1 += tmp;
        }
    }

    tmp = scalar_p1 / scalar_p2; //Дробь
    //Новый x_k

    for (int i = 0; i < size; i++)
    {
        x_k[i] = x_k[i] - tmp * s_k[i];
    }




    int Iteration = 0;
    do {
        Iteration++;

        //скалярное произведение знаменатель  beta_znam
        chunk = 1000;
        beta_znam = 0;
#pragma omp parallel shared(beta_znam) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) reduction(+:beta_znam) nowait
            for (int i = 0; i < size; i++) {
                tmp = f_k[i] * f_k[i];
                beta_znam += tmp;
            }
        }

        //Вычисление градиента f
        chunk = 50;
#pragma omp parallel shared(f_k) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < A.N; i++)
            {
                tmp = 0;
                for (int j = 0; j < A.N; j++) {
                    auto el = A[i][j];

                    tmp += el * x_k[j];


                }
                f_k[i] = tmp - b[i];
            }
        }


        //Вычисление вектора направления
        chunk = 1000;
        scalar_p1 = 0;
        scalar_p2 = 0;
        //скалярное произведение числитель  scalar_p1
#pragma omp parallel shared(scalar_p1) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) reduction(+:scalar_p1) nowait
            for (int i = 0; i < size; i++) {
                tmp = f_k[i] * f_k[i];
                scalar_p1 += tmp;
            }
        }



        tmp = scalar_p1 / beta_znam; //сама дробь

        //Вектор направления s_k
        for (int i = 0; i < size; i++)
        {
            s_k[i] = -f_k[i] + (tmp * s_k[i]);
        }


        //Вычисление смещения величины
        scalar_p1 = 0;
        scalar_p2 = 0;

        //Знаменатель Матрица на вектор
        chunk = 50;
#pragma omp parallel shared(tmp_vec) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < A.N; i++)
            {
                tmp = 0;
                for (int j = 0; j < A.N; j++) {
                    auto el = A[i][j];

                    tmp += el * s_k[j];


                }
                tmp_vec[i] = tmp;
            }
        }

        // Знаменатель скалярное
        chunk = 1000;
#pragma omp parallel shared(scalar_p2) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) reduction(+:scalar_p2) nowait
            for (int i = 0; i < size; i++) {
                tmp = s_k[i] * tmp_vec[i];
                scalar_p2 += tmp;
            }
        }


        //Числитель

#pragma omp parallel shared(scalar_p1) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) reduction(+:scalar_p1) nowait
            for (int i = 0; i < size; i++) {
                tmp = f_k[i] * s_k[i];
                scalar_p1 += tmp;
            }
        }

        tmp = scalar_p1 / scalar_p2; //Дробь
        //Новый x_k

        for (int i = 0; i < size; i++)
        {
            x_k[i] = x_k[i] - tmp * s_k[i];
        }
        std::cout << "\n";
        for (int i = 0; i < size; i++)
        {
            std::cout << x_k[i] << "\n";
        }
    }while (Iteration < 10000);

    std::cout << "\n";
    for (int i = 0; i < size; i++)
    {
        std::cout << x_k[i] << "\n";
    }
    
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
