import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def f(x1, x2):
    return x1 ** 2 + x1 * x2 + 2 * x2 ** 2 - 7 * x1 - 7 * x2


def grad_f(x1, x2):
    df_dx1 = 2 * x1 + x2 - 7
    df_dx2 = x1 + 4 * x2 - 7
    return np.array([df_dx1, df_dx2])


def fletcher_reeves(start_point, alpha=0.1, epsilon=1e-6,
                    max_iterations=100):
    x_current = np.array(start_point)
    gradient = grad_f(*x_current)
    direction = -gradient
    iteration = 0

    # Для хранения истории оптимизации
    history = [x_current.copy()]
    f_values = [f(*x_current)]
    grad_norms = [np.linalg.norm(gradient)]
    alphas = [alpha]
    betas = [0]  # На первой итерации beta=0

    print("\n" + "=" * 80)
    print(
        f"{'Итерация':<10} {'Точка (x1, x2)':<25} {'f(x1,x2)':<15} {'Норма градиента':<20} {'alpha':<10} {'beta':<10}")
    print("=" * 80)

    while np.linalg.norm(gradient) > epsilon and iteration < max_iterations:
        iteration += 1

        x_next = x_current + alpha * direction
        new_gradient = grad_f(*x_next)
        beta = np.dot(new_gradient, new_gradient) / np.dot(gradient,
                                                           gradient)
        direction = -new_gradient + beta * direction

        print(
            f"{iteration:<10} ({x_current[0]:.6f}, {x_current[1]:.6f})  {f(*x_current):<15.6f}  {np.linalg.norm(gradient):<20.6f}  {alpha:<10.4f}  {beta:<10.6f}")

        # Сохраняем историю
        history.append(x_next.copy())
        f_values.append(f(*x_next))
        grad_norms.append(np.linalg.norm(new_gradient))
        alphas.append(alpha)
        betas.append(beta)

        x_current = x_next
        gradient = new_gradient

        if np.linalg.norm(gradient) < epsilon:
            print(
                "=" * 80 + f"\nСходимость достигнута на итерации {iteration}.")
            break

    return x_current, f(
        *x_current), iteration, history, f_values, grad_norms, alphas, betas


def visualize_fletcher_reeves(history, f_values, grad_norms, alphas, betas):
    history = np.array(history)

    # Создаем сетку для графиков функции
    x = np.linspace(0, 6, 100)
    y = np.linspace(0, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # 1. 3D график функции с траекторией
    fig = plt.figure(figsize=(20, 6))

    ax1 = fig.add_subplot(141, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.7)
    ax1.plot(history[:, 0], history[:, 1], f_values, 'r.-', markersize=10,
             linewidth=2)
    ax1.set_title('Метод Флетчера-Ривса (3D)')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x1,x2)')

    # 2. Контурный график с траекторией
    ax2 = fig.add_subplot(142)
    ax2.contour(X, Y, Z, 50, cmap=cm.coolwarm)
    ax2.plot(history[:, 0], history[:, 1], 'r.-', markersize=10,
             linewidth=2)
    ax2.plot(history[0, 0], history[0, 1], 'bo', label='Начальная точка')
    ax2.plot(history[-1, 0], history[-1, 1], 'go', label='Конечная точка')
    ax2.set_title('Траектория поиска')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.legend()

    # 3. Графики сходимости
    ax3 = fig.add_subplot(143)
    iterations = np.arange(len(f_values))
    ax3.plot(iterations, f_values, 'b-', label='Значение функции')
    ax3.set_xlabel('Итерация')
    ax3.set_ylabel('f(x)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')

    ax3b = ax3.twinx()
    ax3b.plot(iterations, grad_norms, 'r--', label='Норма градиента')
    ax3b.set_ylabel('||∇f||', color='r')
    ax3b.tick_params(axis='y', labelcolor='r')

    ax3.set_title('Сходимость алгоритма')

    # 4. График параметров alpha и beta
    ax4 = fig.add_subplot(144)
    ax4.plot(iterations, alphas, 'g-', label='alpha (шаг)')
    ax4.plot(iterations, betas, 'm--', label='beta (параметр)')
    ax4.set_xlabel('Итерация')
    ax4.set_ylabel('Значение параметра')
    ax4.set_title('Параметры метода')
    ax4.legend()

    plt.tight_layout()
    plt.show()


# Запуск алгоритма
start_point = (5.0, 5.0)
min_point, min_value, total_iterations, history, f_values, grad_norms, alphas, betas = fletcher_reeves(
    start_point)

# Вывод результатов
print(f"\nРезультат:")
print(f"Минимальная точка: ({min_point[0]:.6f}, {min_point[1]:.6f})")
print(f"Минимальное значение f(x1, x2): {min_value:.6f}")
print(f"Всего итераций: {total_iterations}")

# Визуализация
visualize_fletcher_reeves(history, f_values, grad_norms, alphas, betas)
