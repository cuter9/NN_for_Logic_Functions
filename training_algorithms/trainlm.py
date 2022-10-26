import numpy as np
import torch
from utilities import plotly_perf, plot_perf


def trainlm(nn, inputs_b, targets_b):  # the vector and matrix form ars same as the ppt

    perf_goal = nn.goal

    mu_p = nn.mu
    mu_pwin_con = nn.mu_pwin
    mu_pwin_div = nn.mu_pwin
    perf_batch_0 = 10e100
    perf_batch = np.empty([1])

    e_sens_0 = torch.zeros((nn.batch_size, nn.batch_size, nn.numOutput[0], nn.numOutput[0]),
                           dtype=torch.float64)
    for i, j in np.ndindex((nn.batch_size, nn.batch_size)):
        if i == j:
            e_sens_0[i, j, :, :] = - torch.diag(torch.ones(nn.numOutput[0], dtype=torch.float64))
    e_sens_0 = torch.transpose(e_sens_0, 1, 2)
    # e_sens_0.transpose(1, 2)

    for i in range(nn.epochs):
        print('epoch:', i, '/', nn.epochs)
        for inputs, targets in zip(inputs_b, targets_b):
            outputs = nn.inference(inputs)
            loss = targets - outputs
            dW, db = backpropagation_lm(nn, loss, e_sens_0)

            # perf_batch_1 = np.sqrt(np.mean(np.square(loss)))
            perf_batch_1 = torch.norm(loss).detach().numpy()
            perf_batch = np.hstack((perf_batch, perf_batch_1))
            d_perf = perf_batch_1 - perf_batch_0
            perf_batch_0 = perf_batch_1

            if perf_batch_1 <= perf_goal:
                break

            if d_perf <= 0:
                mu_pwin_div = nn.mu_pwin
                mu_pwin_con -= 1
                if mu_pwin_con == 0:
                    nn.mu *= nn.mu_dr
                    mu_pwin_con = nn.mu_pwin
            else:
                mu_pwin_con = nn.mu_pwin
                mu_pwin_div -= 1
                if mu_pwin_div == 0:
                    nn.mu *= nn.mu_ir
                    mu_pwin_div = nn.mu_pwin

            mu_p = np.hstack((mu_p, nn.mu))
            nn.W = [1 * (l_W - l_dW) for l_W, l_dW in zip(nn.W, dW)]
            nn.b = [1 * (l_b - l_db) for l_b, l_db in zip(nn.b, db)]

    # plot train convergence
    perf_batch = perf_batch[1:].reshape((-1, nn.total_batch))
    fig_perf = dict(fig_title='Training Convergence (Levenberg-Marquardt Method)',
                    y_scale='log',
                    file_name='Training Convergence_' + nn.train_method + nn.struc_nn + '.html',
                    y_label='mean square error',
                    x_label='epochs',
                    figsize=(16, 10))
    # plotly_perf(perf, fig_perf)
    perf_epoch = np.sqrt(np.mean(np.square(perf_batch), axis=1))
    plotly_perf(perf_epoch, fig_perf)
    # plot_perf(perf_epoch, fig_perf)
    perf = {'perf_batch': perf_batch, 'perf_epoch': perf_epoch, 'stop_epoch': i}

    # plot mu adaptation
    fig_mu = dict(fig_title='Mu Adaptation Profile (Levenberg-Marquardt Method)',
                  y_scale='linear',
                  file_name='mu adaptation_' + nn.train_method + nn.struc_nn + '.html',
                  y_label='mu value',
                  x_label='epochs',
                  figsize=(16, 10))

    mu_p = mu_p[1:].reshape((-1, nn.total_batch))
    plotly_perf(mu_p[:, -1], fig_mu)
    # plot_perf(mu_p[:, -1], fig_mu)

    return perf


def backpropagation_lm(nn, error, e_sens_top):  # use pytorch

    jacob = calc_jacob(nn, e_sens_top)
    error = error[np.newaxis, np.newaxis, :, :]

    jwt = [jcb[1] for jcb in jacob[:]]  # jcb[0]: yh; jcb[1]: weight, jcb[2]: bias
    jbt = [jcb[2] for jcb in jacob[:]]
    jw_f = [torch.reshape(jw_i, (jw_i.shape[0], jw_i.shape[1], -1)) for jw_i in jwt]  #
    jb_f = [torch.reshape(jb_i, (jb_i.shape[0], jb_i.shape[1], -1)) for jb_i in jbt]
    jwb = [torch.cat((jw_i, jb_i), 2) for jw_i, jb_i in zip(jw_f, jb_f)]
    hwb = [torch.transpose(jwb_i, 1, 2) @ jwb_i for jwb_i in
           jwb]  # transpose to form the jacobean in last two dimension
    hwb = [torch.sum(hwb_i, 0) for hwb_i in hwb]  # sum up the hessian of all samples
    jwbe = [tensor_product_draft(error.clone().detach(),
                                        jwb_i[:, :, np.newaxis, :].clone().detach()) for jwb_i in jwb]
    # jwbe = [tensor_product_np(error.clone().detach().numpy(),
    #                                  jwb_i[:, :, np.newaxis, :].clone().detach().numpy()) for jwb_i in jwb]
    # jwbe = [tensor_product(error, jwb_i[:, :, np.newaxis, :]) for jwb_i in jwb]
    jwbe = [jwbe_i[0, 0, :, :] for jwbe_i in jwbe]
    dwb = [jwbe_i @ torch.inverse(hwb_i + nn.mu * torch.eye(hwb_i.shape[0])) for hwb_i, jwbe_i
           in zip(hwb, jwbe)]
    dwb.reverse()
    dwb_split = [torch.split(dwb_i, ws_i[0] * ws_i[1], 1) for dwb_i, ws_i in zip(dwb, nn.weight_shapes)]
    db = [dwb_i[1] for dwb_i in dwb_split]
    dw = [torch.reshape(dwb_i[0], ws_i) for dwb_i, ws_i in zip(dwb_split, nn.weight_shapes)]

    return dw, db


def backpropagation_lm_np(nn, jacob, err):  # us numpy
    dw = []
    db = []
    jTj = []
    e_jacob = []
    for l_jacob in jacob:  # jacobian of lth layer; calculate JTJ and j*err of each layer
        jTj.append(np.sum(
            l_jacob_i.T @ l_jacob_i for l_jacob_i in l_jacob))  # the jacobian sum of all samples
        je = np.sum(l_jacob_i.T @ err_i.T for l_jacob_i, err_i in zip(l_jacob, err))
        e_jacob.append(je)  # gradient sum of all samples

    for l_jTj, le_jacob, lw_shape, lb_shape in zip(jTj, e_jacob, nn.weight_shapes,
                                                   nn.bias_shapes):  # using avove, cal weights adjustment
        ndim = l_jTj.shape[0]

        try:
            l_lm = l_jTj + nn.mu * np.identity(ndim)
        except "Singular matrix":
            l_lm = nn.mu * np.identity(ndim)

        dwb_l = np.linalg.inv(l_lm) @ le_jacob
        # split out the bias weights
        db_l = np.reshape(dwb_l[-lb_shape[0]:], lb_shape)
        db.append(db_l)
        dw_l = np.reshape(dwb_l[0:-lb_shape[0]], lw_shape)
        dw.append(dw_l)
    return dw, db


def calc_jacob_np(nn, sensgrad,
                  inputs):  # use numpy to calculate the jacobian in a list with element being layer
    jacob = []
    yh = [np.hstack((inputs, np.ones((inputs.shape[0], 1))))]
    for yh_i in nn.yh[0:-1]:
        yh_i = np.hstack((yh_i, np.ones((yh_i.shape[0], 1))))
        yh.append(yh_i)

    for l_sgrad, l_yh in zip(sensgrad, yh):  # for each layer
        # transfer each row of the array l_yhg for to list form in row vector form
        jacob_i = []
        for l_sgrad_i, l_yh_i in zip(l_sgrad, l_yh):
            aa = []
            for l_yh_ij in l_yh_i:
                aa.append(l_sgrad_i * l_yh_ij)
            jacob_ij = np.hstack(tuple(aa))
            jacob_i.append(jacob_ij)  # for each ith samples in the layer
        jacob.append(jacob_i)
    return jacob


# use pytorch to calculate the jacobian wrt layer input, W, and b place it in a list with element being layer
def calc_jacob(nn, e_sens_top):
    e_sens = e_sens_top
    jacob = []
    for yhi_1, Wi, bi in zip(reversed(nn.yh[0:-1]), reversed(nn.W), reversed(nn.b)):
        jt = torch.autograd.functional.jacobian(nn.perceptron, (yhi_1, Wi, bi))
        j_layer = [tensor_product_draft(e_sens.clone().detach(), jt_i.clone().detach()) for jt_i
                   in jt]
        e_sens = j_layer[0]
        jacob = jacob + [j_layer]

    return jacob


def calc_hess(nn, inputs):  # used pytorch to calculate the hessian in a list with element being layer
    hess = torch.autograd.functional.hessian(nn.inference, inputs)
    return hess


def calc_sensgrad_np(nn, dedy):  # use numpy
    sensgrad = [[dedy @ np.diagflat(yhg) for yhg in
                 nn.yh_grad[-1]]]  # create a list not array for the first sensitivity at output layer
    W, yh_grad = nn.W, nn.yh_grad  # the gradient = I for softmax output layer
    for i in range(nn.deepness):  # calculate the sensitivity of each layer
        yh_grad_diag = [np.diagflat(yhg) for yhg in yh_grad[-2 - i]]
        sensgrad.append([l_sgrad @ W[-1 - i].T @ l_yhgd for l_yhgd, l_sgrad in zip(yh_grad_diag, sensgrad[-1])])
    sensgrad.reverse()
    return sensgrad


def calc_sensgrad(nn, dedy):  # use pytorch
    sensgrad = [[dedy @ np.diagflat(yhg) for yhg in
                 nn.yh_grad[-1]]]  # create a list not array for the first sensitivity at output layer
    W, yh_grad = nn.W, nn.yh_grad  # the gradient = I for softmax output layer
    for i in range(nn.deepness):  # calculate the sensitivity of each layer
        yh_grad_diag = [np.diagflat(yhg) for yhg in yh_grad[-2 - i]]
        sensgrad.append([l_sgrad @ W[-1 - i].T @ l_yhgd for l_yhgd, l_sgrad in zip(yh_grad_diag, sensgrad[-1])])
    sensgrad.reverse()
    return sensgrad


def tensor_product(a, b):  # calculate matrix_like tensor product
    c = torch.zeros((a.shape[0], a.shape[1], b.shape[2], b.shape[3]))
    for i, j in np.ndindex((a.shape[0], a.shape[1])):
        for m, n in np.ndindex((b.shape[2], b.shape[3])):
            c[i, j, m, n] = torch.sum(a[i, j, :, :] * b[:, :, m, n])

    return c


def tensor_product_draft(a, b):  # calculate matrix_like tensor product

    a1 = torch.reshape(a, (a.shape[0], a.shape[1], 1, -1))
    a1 = torch.squeeze(a1, 2)
    a1 = torch.reshape(a1, (1, -1, a1.shape[2]))
    a1 = torch.squeeze(a1, 0)

    b1 = torch.reshape(b, (1, -1, b.shape[2], b.shape[3]))
    b1 = torch.squeeze(b1, 0)
    b1 = torch.reshape(b1, (b1.shape[0], 1, -1))
    b1 = torch.squeeze(b1, 1)
    # c = torch.zeros((a.shape[0], a.shape[1], b.shape[2], b.shape[3]))
    c1 = torch.matmul(a1, b1)
    c1 = c1[:, np.newaxis, :, np.newaxis]
    c1 = torch.reshape(c1, (a.shape[0], -1, c1.shape[2], c1.shape[3]))
    c1 = torch.reshape(c1, (c1.shape[0], c1.shape[1], b.shape[2], -1))

    return c1


def tensor_product_np(a, b):  # calculate matrix_like tensor product
    c = np.zeros((a.shape[0], a.shape[1], b.shape[2], b.shape[3]))
    for i, j in np.ndindex((a.shape[0], a.shape[1])):
        for m, n in np.ndindex((b.shape[2], b.shape[3])):
            c[i, j, m, n] = np.sum(a[i, j, :, :] * b[:, :, m, n])

    return torch.tensor(c, dtype=torch.float32)
