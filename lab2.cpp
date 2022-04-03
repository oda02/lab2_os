#include <iostream>
#include <cstdlib>
#include <fstream>
#include <omp.h>
#include <Windows.h>

#define TOCHN 0.001

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
double grad_wiki(MyMatrix A, double* b);

MyMatrix& readFile()
{
    
    std::ifstream file("bcsstm34.mtx");
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
{
    
    MyMatrix matrix = readFile();
    double* b = new double[matrix.N];
    for (size_t i = 0; i < matrix.N; i++)
    {
        b[i] = std::rand();
    }


    /*MyMatrix matrix(3);
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
    b[2] = 2;*/
    std::ofstream out;
    double time;

    omp_set_dynamic(0);
    omp_set_num_threads(1);
    
    time = grad_wiki(matrix, b);
    out.open("C:\\Users\\atylt\\Desktop\\res_os.txt", std::ios_base::app);
    out << "\n   1 th \n";
    out << time << std::endl;
    out.close();
    Sleep(300000);

    std::cout << "   2 th \n";
    omp_set_num_threads(2);
    time = grad_wiki(matrix, b);
    out.open("C:\\Users\\atylt\\Desktop\\res_os.txt", std::ios_base::app);
    out << "\n   2 th \n";
    out << time << std::endl;
    out.close();
    Sleep(300000);

    std::cout << "   4 th \n";
    omp_set_num_threads(4);
    time = grad_wiki(matrix, b);
    out.open("C:\\Users\\atylt\\Desktop\\res_os.txt", std::ios_base::app);
    out << "\n   4 th \n";
    out << time << std::endl;
    out.close();
    Sleep(300000);

    std::cout << "   8 th \n";
    omp_set_num_threads(8);
    time = grad_wiki(matrix, b);
    out.open("C:\\Users\\atylt\\Desktop\\res_os.txt", std::ios_base::app);
    out << "\n   8 th \n";
    out << time << std::endl;
    out.close();
    Sleep(300000);
    
    std::cout << "   16 th \n";
    omp_set_num_threads(16);
    time = grad_wiki(matrix, b);
    out.open("C:\\Users\\atylt\\Desktop\\res_os.txt", std::ios_base::app);
    out << "\n   16 th \n";
    out << time << std::endl;
    out.close();
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

double grad_wiki(MyMatrix A, double* b)
{
    double min = 99999;
    int size = A.N;
    double start = omp_get_wtime();
    double* x_k = new double[size];
    double* r_k = new double[size];
    double* z_k = new double[size];

    double* tmp_vec = new double[size];

    double tmp, scalar_p1, scalar_p2, beta_znam, a_k, b_k, b_norm;
    int chunk = 1000;

    b_norm = 0;

    //Начальное приближение
    std::fill(x_k, x_k + size, 0.);
    for (size_t i = 0; i < size; i++)
    {
        z_k[i] = r_k[i] = b[i];
        //z_k[i] = b[i];
    }

    //Норма b
#pragma omp parallel shared(b_norm) private(tmp)
    {
#pragma omp for schedule(dynamic, chunk) reduction(+:b_norm) nowait
        for (int i = 0; i < size; i++) {
            tmp = b[i] * b[i];
            b_norm += tmp;
        }
    }
    
    int Iteration = 0;
    do {
        Iteration++;

        //Вычисляем a_k


        //Вычисление A * z_k-1
        chunk = 50;
#pragma omp parallel shared(z_k) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < A.N; i++)
            {
                tmp = 0;
                for (int j = 0; j < A.N; j++) {
                    tmp += A[i][j] * z_k[j];
                }
                tmp_vec[i] = tmp;
            }
        }

        //скалярное произведение r_k-1 * r_k-1
        chunk = 1000;
        scalar_p1 = scalar_p2 = 0;
#pragma omp parallel shared(scalar_p1) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) reduction(+:scalar_p1) nowait
            for (int i = 0; i < size; i++) {
                tmp = r_k[i] * r_k[i];
                scalar_p1 += tmp;
            }
        }

        //Вычисление знаметаеля tmp_vec * z_k-1
#pragma omp parallel shared(scalar_p2) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) reduction(+:scalar_p2) nowait
            for (int i = 0; i < size; i++) {
                tmp = tmp_vec[i] * r_k[i];
                scalar_p2 += tmp;
            }
        }

        a_k = scalar_p1 / scalar_p2;

        //Вычисление x_k

#pragma omp parallel shared(r_k) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < size; i++) {
                x_k[i] = x_k[i] + a_k * z_k[i];
            }
        }

        //for (int i = 0; i < size; i++)
        //{
        //    x_k[i] = x_k[i] + a_k * z_k[i];
        //}

        scalar_p1 = scalar_p2 = 0;

        //Вычисляем beta
        //скалярное произведение знаменатель  scalar_p2 для beta
#pragma omp parallel shared(scalar_p2) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) reduction(+:scalar_p2) nowait
            for (int i = 0; i < size; i++) {
                tmp = r_k[i] * r_k[i];
                scalar_p2 += tmp;
            }
        }

        //Вычисление r_k
        
        
        for (int i = 0; i < size; i++)
        {
            r_k[i] = r_k[i] - a_k * tmp_vec[i];
        }
        

       
 
        //скалярное произведение числитель  scalar_p1
#pragma omp parallel shared(scalar_p1) private(tmp)
        {
#pragma omp for schedule(dynamic, chunk) reduction(+:scalar_p1) nowait
            for (int i = 0; i < size; i++) {
                tmp = r_k[i] * r_k[i];
                scalar_p1 += tmp;
            }
        }





        b_k = scalar_p1 / scalar_p2; //сама дробь


        for (int i = 0; i < size; i++)
        {
            z_k[i] = r_k[i] + b_k * z_k[i];
        }
        
        //проверяем невязку

        //Норма r_k = scalar_p2


        //tmp = scalar_p2 / b_norm;
        //std::cout << tmp << "   " << min << "\n";
        /*if (tmp < min)
        {
            std::cout << tmp << "\n";
            min = tmp;
        }*/
        //std::cout << scalar_p2 / b_norm << "\n";

        /*for (int i = 0; i < size; i++)
        {
            std::cout << x_k[i] << "\n";
        }*/
    } while (TOCHN < (scalar_p2 / b_norm));

    /*std::cout << "\n";
    for (int i = 0; i < size; i++)
    {
        std::cout << x_k[i] << "\n";
    }*/
    std::cout << "\n " << Iteration;
    return (omp_get_wtime() - start);

}
