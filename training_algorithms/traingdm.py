import torch
import numpy as np
from utilities import plotly_perf, plot_perf


def traingdm(nn, inputs_b, targets_b):
    perf_goal = nn.goal

    dW0 = [torch.zeros(s, requires_grad=True) for s in nn.weight_shapes]
    db0 = [torch.zeros(s, requires_grad=True) for s in nn.bias_shapes]
    perf_batch = np.array(0)
    for i in range(nn.epochs):
        for inputs, targets in zip(inputs_b, targets_b):
            # outputs = nn.inference_softmax(inputs)
            outputs = nn.inference(inputs)

            error = targets - outputs
            loss = error.pow(2).sum()
            dW, db = backpropagation(nn, loss)

            dW0 = [dW_t + nn.mem * dW0_t for dW_t, dW0_t in zip(dW, dW0)]
            nn.W = [(W_t + dW0_t).clone().detach().requires_grad_(True) for W_t, dW0_t in zip(nn.W, dW0)]
            # nn.W = [torch.tensor(W_t + dW0_t, requires_grad=True) for W_t, dW0_t in zip(nn.W, dW0)]

            db0 = [db_t + nn.mem * db0_t for db_t, db0_t in zip(db, db0)]
            nn.b = [(b_t + db0_t).clone().detach().requires_grad_(True) for b_t, db0_t in zip(nn.b, db0)]
            # nn.b = [torch.tensor(b_t + db0_t, requires_grad=True) for b_t, db0_t in zip(nn.b, db0)]

            perf = torch.norm(error).detach().numpy()
            perf_batch = np.hstack((perf_batch, perf))
            if perf <= perf_goal:
                break

        # print('Training error at {0}th epoch : {1}'.format(i, np.sqrt(np.mean(np.square(perf[i, :])))))

    perf_batch = perf_batch[1:].reshape((-1, nn.total_batch))
    # perf_epoch = np.sqrt(np.mean(np.square(perf_batch), axis=1))
    perf_epoch = np.linalg.norm(perf_batch, axis=1)

    fig_perf = dict(fig_title='Training Convergence (gradient descent method)',
                    y_scale='log',
                    file_name='Training Convergence_' + nn.train_method + nn.struc_nn + '.html',
                    y_label='mean square error',
                    x_label='epochs',
                    figsize=(16, 10))

    # plot_perf(perf_epoch, fig_perf)
    plotly_perf(perf_epoch, fig_perf)
    perf = {'perf_batch': perf_batch, 'perf_epoch': perf_epoch, 'stop_epoch': i}
    return perf


def backpropagation_np(nn, err, inputs):
    dw = []
    db = []
    errgrad = calc_errgrad(err)
    yh = [inputs] + nn.yh[:-1]
    b_shape = nn.Layers[1:]

    for yh_t, errgrad_t, lb_shape in zip(yh, errgrad, b_shape):
        dw.append(nn.lr * (yh_t.T @ errgrad_t))
        db.append(np.reshape(nn.lr * np.sum(errgrad_t, axis=0), (lb_shape, 1)))
    return dw, db


def backpropagation(nn, loss):
    torch.autograd.backward(loss, retain_graph=True)
    dW = [-nn.lr * Wt.grad for Wt in nn.W]
    db = [-nn.lr * bt.grad for bt in nn.b]

    return dW, db


def calc_errgrad(nn, err):  # use err = identity matrix when calculate sensitivity propagation
    errgrad = [nn.yh_grad[-1] * err]
    W, yh_grad = nn.W, nn.yh_grad
    for i in range(nn.deepness):
        errgrad.append(yh_grad[-2 - i] * (errgrad[-1] @ W[-1 - i].T))
    errgrad.reverse()
    return errgrad

    # the think method returns both the values for
