"""
Solve Linear Programming (LP) problem using Neuromancer:
minimize     a1*x-a2*y
subject to   x+y-p1 >= 0
             -2*x+y+p2 >= 0
             x-2*y+p3 >= 0
problem parameters:            a1, a2, p1, p2, p3
problem decition variables:    x, y
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
#from matplotlib.patheffects import withTickedStroke

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import Variable, Objective, Loss
from neuromancer.activations import activations
from neuromancer import policies
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import normalize_data, split_static_data, StaticDataset
from neuromancer.plot import plot_loss_mpp, plot_solution_mpp


def arg_mpLP_problem(prefix=''):
    """
    Command line parser for mpLP problem definition arguments
    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("mpLP")
    gp.add("-Q", type=float, default=1.0,
           help="loss function weight.")  # tuned value: 1.0
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_con", type=float, default=2.0,
           help="constraints penalty weight.")  # tuned value: 50.0
    gp.add("-nx_hidden", type=int, default=40,
           help="Number of hidden states of the solution map")
    gp.add("-n_layers", type=int, default=2,
           help="Number of hidden layers of the solution map")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=800,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.0001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=200,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=200,
           help="Number of epochs to wait before enacting early stopping policy.")
    return parser


def get_dataloaders(data, norm_type=None, split_ratio=None, num_workers=0):
    """This will generate dataloaders for a given dictionary of data.
    Dataloaders are hard-coded for full-batch training to match NeuroMANCER's training setup.
    :param data: (dict str: np.array or list[dict str: np.array]) data dictionary or list of data
        dictionaries; if latter is provided, multi-sequence datasets are created and splits are
        computed over the number of sequences rather than their lengths.
    :param norm_type: (str) type of normalization; see function `normalize_data` for more info.
    :param split_ratio: (list float) percentage of data in train and development splits; see
        function `split_sequence_data` for more info.
    """

    if norm_type is not None:
        data, _ = normalize_data(data, norm_type)
    train_data, dev_data, test_data = split_static_data(data, split_ratio)

    train_data = StaticDataset(
        train_data,
        name="train",
    )
    dev_data = StaticDataset(
        dev_data,
        name="dev",
    )
    test_data = StaticDataset(
        test_data,
        name="test",
    )

    train_data = DataLoader(
        train_data,
        batch_size=len(train_data),
        shuffle=False,
        collate_fn=train_data.collate_fn,
        num_workers=num_workers,
    )
    dev_data = DataLoader(
        dev_data,
        batch_size=len(dev_data),
        shuffle=False,
        collate_fn=dev_data.collate_fn,
        num_workers=num_workers,
    )
    test_data = DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
        collate_fn=test_data.collate_fn,
        num_workers=num_workers,
    )

    return (train_data, dev_data, test_data), train_data.dataset.dims


if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpLP_problem()])
    args, grps = parser.parse_arg_groups()
    args.bias = True
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    """
    # # #  Dataset 
    """
    #  randomly sampled parameters theta generating superset of:
    #  theta_samples.min() <= theta <= theta_samples.max()
    np.random.seed(args.data_seed)
    nsim = 10000  # number of datapoints: increase sample density for more robust results
    samples = {"a1": np.random.uniform(low=0.1, high=1.5, size=(nsim, 1)),
               "a2": np.random.uniform(low=0.1, high=2.0, size=(nsim, 1)),
               "p1": np.random.uniform(low=5.0, high=10.0, size=(nsim, 1)),
               "p2": np.random.uniform(low=5.0, high=10.0, size=(nsim, 1)),
               "p3": np.random.uniform(low=5.0, high=10.0, size=(nsim, 1))}
    nstep_data, dims = get_dataloaders(samples)
    train_data, dev_data, test_data = nstep_data

    """
    # # #  mpLP problem formulation in Neuromancer
    """
    n_var = 2           # number of decision variables
    # define solution map as MLP policy
    # defining expected dimensions of the solution variable: internal policy key 'U'
    dims['U'] = (nsim, n_var)
    activation = activations['relu']
    linmap = slim.maps['linear']
    sol_map = policies.MLPPolicy(
        {**dims},
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=["a1", "a2", "p1", "p2", "p3"],
        name='sol_map',
    )

    # variables
    x = Variable(f"U_pred_{sol_map.name}")[:, :, 0]
    y = Variable(f"U_pred_{sol_map.name}")[:, :, 1]
    # sampled parameters
    a1 = Variable('a1')
    a2 = Variable('a2')
    p1 = Variable('p1')
    p2 = Variable('p2')
    p3 = Variable('p3')

    # objective function
    loss = Objective(a1*x+a2*y, weight=args.Q, name='loss')
    # constraints
    con_1 = args.Q_con*(x+y-p1 >= 0)
    con_2 = args.Q_con*(-2*x+y+p2 >= 0)
    con_3 = args.Q_con*(x-2*y+p3 >= 0)

    # constrained optimization problem construction
    objectives = [loss]
    constraints = [con_1, con_2, con_3]
    components = [sol_map]
    model = Problem(objectives, constraints, components)
    model = model.to(device)

    """
    # # # Metrics and Logger
    """
    args.savedir = 'test_mpLP_1'
    args.verbosity = 1
    metrics = ["dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir,
                         verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'mpLP_1'

    """
    # # #  mpLP problem solution in Neuromancer
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # define trainer
    trainer = Trainer(
        model,
        train_data,
        dev_data,
        test_data,
        optimizer,
        logger=logger,
        epochs=args.epochs,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
        patience=args.patience,
        warmup=args.warmup,
        device=device,
    )

    # Train mpLP solution map
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)

    # # # plots
    # plot_loss_mpp(model, train_data, xmin=-2, xmax=2, save_path=None)
    # plot_solution_mpp(sol_map, xmin=-2, xmax=2, save_path=None)

    # # # Plot the solution # # #

    # test problem parameters
    a1 = 0.5
    a2 = 1.0
    p1 = 6.0
    p2 = 8.0
    p3 = 9.0

    x1 = np.arange(-1.0, 10.0, 0.05)
    y1 = np.arange(-1.0, 10.0, 0.05)
    xx, yy = np.meshgrid(x1, y1)

    # eval objective and constraints
    J = a1 * xx + a2 * yy
    c1 = xx + yy - p1
    c2 = -2 * xx + yy + p2
    c3 = xx - 2 * yy + p3

    # Plot
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(xx, yy, J,
                     alpha=0.6)
    fig.colorbar(cp)
    ax.set_title('Linear problem')
    cg1 = ax.contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
    plt.setp(cg1.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)
    cg2 = ax.contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.7)
    plt.setp(cg2.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)
    cg3 = ax.contour(xx, yy, c3, [0], colors='mediumblue', alpha=0.7)
    plt.setp(cg3.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)

    params = torch.tensor([a1, a2, p1, p2, p3])
    xy_optim = model.components[0].net(params).detach().numpy()
    print(xy_optim[0])
    print(xy_optim[1])
    ax.plot(xy_optim[0], xy_optim[1], 'r*', markersize=10)
