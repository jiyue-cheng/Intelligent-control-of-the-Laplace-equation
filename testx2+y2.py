import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from matplotlib import cm
import pandas as pd
from torch.optim import lr_scheduler

device = torch.device("cuda:0")

# 模型搭建
class Net(nn.Module):
    def __init__(self, NN): # NL n个l（线性，全连接）隐藏层， NN 输入数据的维数， 128 256
        # NL是有多少层隐藏层
        # NN是每层的神经元数量
        super(Net, self).__init__()

        self.input_layer = nn.Linear(2, NN)
        self.hidden_layer1 = nn.Linear(NN,int(NN/2))
        self.hidden_layer2 = nn.Linear(int(NN/2), int(NN/2))
        self.output_layer = nn.Linear(int(NN/2), 1)

    def forward(self, x): # 一种特殊的方法 __call__() 回调
        out = torch.tanh(self.input_layer(x))
        out = torch.tanh(self.hidden_layer1(out))
        out = torch.tanh(self.hidden_layer2(out))
        out_final = self.output_layer(out)
        return out_final

def pde(x, net):
    u_hat = net(x).to(device)  # 网络得到的数据
    x.requires_grad_(True).to(device)
    u_xy = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]  # 求偏导数
    d_x = u_xy[:, 0].unsqueeze(-1)
    d_y = u_xy[:, 1].unsqueeze(-1)

    u_xx = torch.autograd.grad(d_x, x, grad_outputs=torch.ones_like(d_x),
                               create_graph=True, allow_unused=True)[0][:, 0].unsqueeze(-1)  # 求偏导数
    u_yy = torch.autograd.grad(d_y, x, grad_outputs=torch.ones_like(d_y),
                               create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)  # 求偏导数

    return u_xx+u_yy  # 公式（1）


# torch.manual_seed(123)
# 初始化傅立叶系数和频率参数
a_nm = torch.zeros((6, 6), requires_grad=True)
# print(a_nm)
b_nm = torch.zeros((6, 6), requires_grad=True)
c_nm = torch.rand((6, 6), requires_grad=True)
d_nm = torch.rand((6, 6), requires_grad=True)
wx = torch.tensor(1.0, requires_grad=True)
wy = torch.tensor(1.0, requires_grad=True)
with torch.no_grad():
    b_nm[0] = torch.zeros(b_nm.size(1))
    c_nm[:, 0] = torch.zeros(c_nm.size(0))
    d_nm[0] = torch.zeros(d_nm.size(1))
    d_nm[:, 0] = torch.zeros(d_nm.size(0))

# 定义傅立叶逼近函数，cos-cos,cos-sin,sin-cos,sin-sin
def u_approximate(x, y, a_nm, b_nm, c_nm, d_nm, wx, wy):
    sum_cos_cos = sum(a_nm[n, m] * torch.cos(n * wx * x) * torch.cos(m * wy * y)
                      for n in range(6) for m in range(6))
    sum_sin_cos = sum(b_nm[n, m] * torch.sin(n * wx * x) * torch.cos(m * wy * y)
                      for n in range(1, 6) for m in range(6))
    sum_cos_sin = sum(c_nm[n, m] * torch.cos(n * wx * x) * torch.sin(m * wy * y)
                      for n in range(6) for m in range(1, 6))
    sum_sin_sin = sum(d_nm[n, m] * torch.sin(n * wx * x) * torch.sin(m * wy * y)
                      for n in range(1, 6) for m in range(1, 6))
    return sum_cos_cos + sum_sin_cos + sum_cos_sin + sum_sin_sin


net = Net(30)
net = net.to(device)
mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# 定义外部参数和它们的优化器
parameters_to_optimize = [a_nm, b_nm, c_nm, d_nm, wx, wy]
initial_lr = 0.005
external_optimizer = torch.optim.Adam(parameters_to_optimize,lr=initial_lr)
# external_scheduler = lr_scheduler.StepLR(external_optimizer, 5, 0.8) # 学习率递减
hist = []   # 存储内部损失
external_hist = []   # 存储外部损失

# 初始化 问题域 矩形域 x∈[-1,1] y∈[0,1]
y_bottom = np.zeros((2000, 1))
y_top = np.ones((2000, 1))
x_right = np.ones((2000, 1))
x_left = -np.ones((2000, 1))
u_in_zeros = np.zeros((2000, 1))

# 定义目标函数
def u_target_fcn(x, y):
    # return x.pow(2) - y.pow(2)
    # return torch.sin(np.pi * x) + torch.sin(np.pi * y)
    # return torch.zeros(25600, 1, dtype=torch.float32)
    return 2*x+3*y

# def adjust_learning_rate(optimizer, loss):
#     lr = initial_lr * (2 * (loss / 20))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


iterations = 500
External_iterations = 1000
for external_epoch in range(External_iterations):
    external_optimizer.zero_grad()
    for epoch in range(iterations):
        optimizer.zero_grad()  # 梯度归0
        # 求边界条件的误差
        # 初始化变量
        x_var = np.random.uniform(low=-1.0, high=1.0, size=(2000, 1))
        y_var = np.random.uniform(low=0, high=1.0, size=(2000, 1))
        # u_bc_sin = u_approximate(x_var, y_var, a_nm, b_nm, c_nm, d_nm, wx, wy)

        # 将数据转化为torch可用的
        pt_x_var = Variable(torch.from_numpy(x_var).float(), requires_grad=True)
        pt_y_var = Variable(torch.from_numpy(y_var).float(), requires_grad=True)
        pt_x_right = Variable(torch.from_numpy(x_right).float(), requires_grad=True)
        pt_x_left = Variable(torch.from_numpy(x_left).float(), requires_grad=True)
        pt_y_bottom = Variable(torch.from_numpy(y_bottom).float(), requires_grad=True)
        pt_y_top = Variable(torch.from_numpy(y_top).float(), requires_grad=True)
        pt_u_in_zeros = Variable(torch.from_numpy(u_in_zeros).float(), requires_grad=True)
        # pt_u_bc_sin = Variable(torch.from_numpy(u_bc_sin).float(), requires_grad=False)
        # pt_u_bc_sin = u_approximate(pt_x_var, pt_y_var, a_nm, b_nm, c_nm, d_nm, wx, wy)
        # print(pt_x_var)
        # print(pt_u_bc_sin)

        # 求边界条件的损失
        net_bc_out1 = net(torch.cat([pt_x_var, pt_y_bottom], 1).to(device))  # u(x,0)的输出
        pt_u_bc_sin1 = u_approximate(pt_x_var, pt_y_bottom, a_nm, b_nm, c_nm, d_nm, wx, wy).to(device)
        net_bc_out2 = net(torch.cat([pt_x_var, pt_y_top], 1).to(device))  # u(x,1)的输出
        pt_u_bc_sin2 = u_approximate(pt_x_var, pt_y_top, a_nm, b_nm, c_nm, d_nm, wx, wy).to(device)
        net_bc_out3 = net(torch.cat([pt_x_right, pt_y_var], 1).to(device))  # u(1,y) 公式（3)
        pt_u_bc_sin3 = u_approximate(pt_x_right, pt_y_var, a_nm, b_nm, c_nm, d_nm, wx, wy).to(device)
        net_bc_out4 = net(torch.cat([pt_x_left, pt_y_var], 1).to(device))  # u(-1,y) 公式（4）
        pt_u_bc_sin4 = u_approximate(pt_x_left, pt_y_var, a_nm, b_nm, c_nm, d_nm, wx, wy).to(device)

        mse_u_2 = mse_cost_function(net_bc_out1, pt_u_bc_sin1).to(device)  # e = u(x,0)-(-sin(pi*x))  公式（2）
        # mse_u_2 = mse_cost_function(net_bc_out, pt_u_in_zeros)  # e = u(x,y)  公式（2）
        mse_u_3 = mse_cost_function(net_bc_out2, pt_u_bc_sin2).to(device)  # e = u(x,1)-(-sin(pi*x))  公式（2）
        mse_u_4 = mse_cost_function(net_bc_out3, pt_u_bc_sin3).to(device)  # e = 0-u(1,y) 公式(3)
        mse_u_5 = mse_cost_function(net_bc_out4, pt_u_bc_sin4).to(device)  # e = 0-u(-1,y) 公式（4）

        # 求PDE函数式的误差
        # 初始化变量
        x_collocation = np.random.uniform(low=-1.0, high=1.0, size=(2000, 1))
        y_collocation = np.random.uniform(low=0.0, high=1.0, size=(2000, 1))
        all_zeros = np.zeros((2000, 1))
        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad=True).to(device)
        pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

        # 将变量x,t带入公式（1）
        f_out = pde(torch.cat([pt_x_collocation, pt_y_collocation], 1).to(device), net).to(device)  # output of f(x,y) 公式（1）
        mse_f_1 = mse_cost_function(f_out, pt_all_zeros).to(device)

        # 将误差(损失)累加起来
        loss = mse_f_1 + mse_u_2 + mse_u_3 + mse_u_4 + mse_u_5
        # if loss < 0.0001:
        #     break
        loss.backward()  # 反向传播
        optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        hist.append(loss.item())  # 保存损失值

        with torch.autograd.no_grad():
            if epoch % 100 == 0:
                print(epoch, "Traning Loss:", loss.data,"PDE LOSS:",mse_f_1,"BC LOSS1:",mse_u_2,"BC LOSS2:",mse_u_3,"BC LOSS3:",mse_u_4,"BC LOSS4:",mse_u_5)
                print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
    x = np.linspace(-1, 1, 256)
    y = np.linspace(0, 1, 100)
    ms_x, ms_y = np.meshgrid(x, y)
    x = np.ravel(ms_x).reshape(-1, 1)
    y = np.ravel(ms_y).reshape(-1, 1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
    pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True)
    pt_u0 = net(torch.cat([pt_x, pt_y], 1).to(device))
    pt_u_target = u_target_fcn(pt_x, pt_y).to(device)
    u0 = pt_u0.data.cpu().numpy()
    u_target = pt_u_target.data.cpu().numpy()
    # 计算平方均差
    external_loss = mse_cost_function(pt_u0, pt_u_target).to(device)
    if external_loss <0.001:
       break


    u = pt_u0.data.cpu().numpy()
    pt_u0 = u.reshape(100, 256)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim([-5, 5])
    # 将张量转换为NumPy数组
    pt_x_np = pt_x.detach().numpy()
    pt_y_np = pt_y.detach().numpy()
    ax.scatter(pt_x_np, np.zeros_like(pt_x_np), u_approximate(pt_x, 0, a_nm, b_nm, c_nm, d_nm, wx, wy).detach().numpy(),color='k', s=20)
    ax.scatter(pt_x_np, np.ones_like(pt_x_np), u_approximate(pt_x, 1, a_nm, b_nm, c_nm, d_nm, wx, wy).detach().numpy(),color='k', s=20)
    ax.scatter(np.ones_like(pt_y_np), pt_y_np, u_approximate(1, pt_y, a_nm, b_nm, c_nm, d_nm, wx, wy).detach().numpy(),color='k', s=20)
    ax.scatter(-np.ones_like(pt_y_np), pt_y_np,u_approximate(-1, pt_y, a_nm, b_nm, c_nm, d_nm, wx, wy).detach().numpy(), color='k', s=20)
    # ax.plot_surface(ms_x, ms_y, pt_u0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.plot_surface(ms_x, ms_y, pt_u0, cmap='gray', edgecolor='black', linewidth=0.0003, antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.savefig('./result_x2-y2/8/pic{}.png'.format(external_epoch + 1))
    external_loss.backward()  # 反向传播计算梯度
    print("这是a_nm梯度：",a_nm.grad)
    external_optimizer.step()  # 更新外部参数
    print("第%d个epoch的学习率：%f" % (external_epoch, external_optimizer.param_groups[0]['lr']))
    # external_scheduler.step()
    # adjust_learning_rate(external_optimizer, external_loss.data)
    external_hist.append(external_loss.item())  # 保存损失值
    df = pd.DataFrame(external_hist)
    df.to_csv('external_loss_2x3y.csv', index=False)

    # 打印信息或其他处理
    with torch.autograd.no_grad():
        if external_epoch % 1 == 0:
            print(external_epoch, "External Loss:", external_loss.data,"\n")
            print(a_nm, b_nm, c_nm, d_nm, wx, wy)

# print(a_nm, b_nm, c_nm, d_nm, wx, wy)
# 打印损失曲线
# print(hist)
# plt.plot(hist)
# plt.xlabel("Update Step")
# plt.ylabel("Loss")
# plt.yscale("log")
# plt.grid(True)
# plt.show()

# 保存损失
# df = pd.DataFrame(external_hist)
# df.to_csv('external_loss_x2+y2.csv',index=False)
# plt.plot(external_hist)
# plt.xlabel("Update Step")
# plt.ylabel("Loss")
# plt.yscale("log")
# plt.grid(True)
# plt.show()


## 画图 ##
x = np.linspace(-1, 1, 256)
y = np.linspace(0, 1, 100)
ms_x, ms_y = np.meshgrid(x, y)
x = np.ravel(ms_x).reshape(-1, 1)
y = np.ravel(ms_y).reshape(-1, 1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True)
pt_u0 = net(torch.cat([pt_x, pt_y], 1).to(device))
u = pt_u0.data.cpu().numpy()
pt_u0 = u.reshape(100, 256)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_zlim([-5, 5])
# 将张量转换为NumPy数组
pt_x_np = pt_x.detach().numpy()
pt_y_np = pt_y.detach().numpy()
ax.scatter(pt_x_np, np.zeros_like(pt_x_np), u_approximate(pt_x, 0, a_nm, b_nm, c_nm, d_nm, wx, wy).detach().numpy(),color='k', s=20)
ax.scatter(pt_x_np, np.ones_like(pt_x_np), u_approximate(pt_x, 1, a_nm, b_nm, c_nm, d_nm, wx, wy).detach().numpy(),color='k', s=20)
ax.scatter(np.ones_like(pt_y_np), pt_y_np, u_approximate(1, pt_y, a_nm, b_nm, c_nm, d_nm, wx, wy).detach().numpy(),color='k', s=20)
ax.scatter(-np.ones_like(pt_y_np), pt_y_np,u_approximate(-1, pt_y, a_nm, b_nm, c_nm, d_nm, wx, wy).detach().numpy(), color='k', s=20)
# ax.plot_surface(ms_x, ms_y, pt_u0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.plot_surface(ms_x, ms_y, pt_u0, cmap='gray', edgecolor='black', linewidth=0.0003, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.savefig('./result_x2-y2/8/pic{}.svg'.format(External_iterations + 1), dpi=300)
# plt.show()
# plt.close(fig)