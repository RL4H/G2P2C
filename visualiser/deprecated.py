from matplotlib import pyplot as plt

def plot_training_logs(dict, dis_len, version=1):

    if version == 1:
        fig = plt.figure(figsize=(16, 26))
        ax = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

        for exp in dict:
            if dict[exp]['show']:
                d1 = dict[exp]['id'].get_training_logs()
                ax.plot(d1['steps'], d1['exp_var'], color=dict[exp]['color'], label=dict[exp]['id'].id)
                ax2.plot(d1['steps'], d1['val_loss'], color=dict[exp]['color'], label=dict[exp]['id'].id)
                ax3.plot(d1['steps'], d1['policy_grad'], color=dict[exp]['color'], label=dict[exp]['id'].id)
                ax4.plot(d1['steps'], d1['value_grad'], color=dict[exp]['color'], label=dict[exp]['id'].id)
        ax.set_title('Explained Variance')
        ax2.set_title('Value Network Loss')
        ax3.set_title('Policy Network Gradients')
        ax4.set_title('Value Network Gradients')

        ax.set_ylabel('Explained Variance')
        ax2.set_ylabel('Loss')
        ax3.set_ylabel('Mean Norm Gradient')
        ax4.set_ylabel('Mean Norm Gradient')

        ax.legend(loc="upper right")
        ax2.legend(loc="upper right")
        ax3.legend(loc="upper right")
        ax4.legend(loc="upper right")

        ax.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)

        ax.set_xlim(0, dis_len)
        ax2.set_xlim(0, dis_len)
        ax3.set_xlim(0, dis_len)
        ax4.set_xlim(0, dis_len)

        #ax2.set_ylim(0, 0.2)
        # ax.set_ylim(-1, 1)
        # ax4.set_ylim(0, 20)
        plt.show()

    elif version == 2:
        fig = plt.figure(figsize=(16, 26))
        ax = fig.add_subplot(511)
        ax_temp = fig.add_subplot(512)
        ax2 = fig.add_subplot(513)
        ax3 = fig.add_subplot(514)
        ax4 = fig.add_subplot(515)

    elif version == 3:
        fig = plt.figure(figsize=(16, 26))
        ax = fig.add_subplot(611)
        ax_temp = fig.add_subplot(612)
        ax2 = fig.add_subplot(613)
        ax3 = fig.add_subplot(614)
        ax4 = fig.add_subplot(615)
        ax5 = fig.add_subplot(616)

        for exp in dict:
            if dict[exp]['show']:
                d1 = dict[exp]['id'].get_training_logs()
                ax.plot(d1['steps'], d1['exp_var'], color=dict[exp]['color'], label=dict[exp]['id'].id)
                ax_temp.plot(d1['steps'], d1['true_var'], color=dict[exp]['color'], label=dict[exp]['id'].id)
                ax2.plot(d1['steps'], d1['val_loss'], color=dict[exp]['color'], label=dict[exp]['id'].id)
                ax3.plot(d1['steps'], d1['policy_grad'], color=dict[exp]['color'], label=dict[exp]['id'].id)
                ax4.plot(d1['steps'], d1['value_grad'], color=dict[exp]['color'], label=dict[exp]['id'].id)
                if version == 3:
                    ax5.plot(d1['steps'], d1['pi_loss'], color=dict[exp]['color'], label=dict[exp]['id'].id)
        ax.set_title('Explained Variance')
        ax_temp.set_title('true returns varience')
        ax2.set_title('Value Network Loss')
        ax3.set_title('Policy Network Gradients')
        ax4.set_title('Value Network Gradients')

        ax.set_ylabel('Explained Variance')
        ax2.set_ylabel('Loss')
        ax3.set_ylabel('Mean Norm Gradient')
        ax4.set_ylabel('Mean Norm Gradient')

        ax.legend(loc="upper right")
        ax_temp.legend(loc="upper right")
        ax2.legend(loc="upper right")
        ax3.legend(loc="upper right")
        ax4.legend(loc="upper right")

        ax.grid(True)
        ax_temp.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)

        ax.set_xlim(0, dis_len)
        ax_temp.set_xlim(0, dis_len)
        ax2.set_xlim(0, dis_len)
        ax3.set_xlim(0, dis_len)
        ax4.set_xlim(0, dis_len)

        if version == 3:
            ax5.grid(True)
            ax5.set_xlim(0, dis_len)
            ax5.legend(loc="upper right")

        # ax2.set_ylim(0, 0.2)
        # ax.set_ylim(-1, 1)
        # ax4.set_ylim(0, 20)

        plt.show()

def plot_aux_training(dict, dis_len, ymax):
    fig = plt.figure(figsize=(16, 18))
    ax = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    for exp in dict:
        if dict[exp]['show']:
            d1 = dict[exp]['id'].get_aux_training_logs()
            ax.plot(d1['steps'], d1['pi_aux_loss'], color=dict[exp]['color'], label=dict[exp]['id'].id)
            ax2.plot(d1['steps'], d1['vf_aux_loss'], color=dict[exp]['color'], label=dict[exp]['id'].id)
            ax3.plot(d1['steps'], d1['pi_aux_grad'], color=dict[exp]['color'], label=dict[exp]['id'].id)
            ax4.plot(d1['steps'], d1['vf_aux_grad'], color=dict[exp]['color'], label=dict[exp]['id'].id)
    ax.set_title('pi_aux_loss')
    ax2.set_title('vf_aux_loss')
    ax3.set_title('pi_aux_grad')
    ax4.set_title('vf_aux_grad')
    ax.set_ylabel('pi_aux_loss')
    ax2.set_ylabel('vf_aux_loss')
    ax3.set_ylabel('pi_aux_grad')
    ax4.set_ylabel('vf_aux_grad')
    ax.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax3.legend(loc="upper right")
    ax4.legend(loc="upper right")
    ax.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax.set_xlim(0, dis_len)
    ax2.set_xlim(0, dis_len)
    # ax.set_ylim(0, ymax)
    # ax2.set_ylim(0, ymax)
    ax3.set_xlim(0, dis_len)
    ax4.set_xlim(0, dis_len)
    plt.show()


def plot_planning_training(dict, dis_len, ymax):
    fig = plt.figure(figsize=(16, 18))
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for exp in dict:
        if dict[exp]['show']:
            d1 = dict[exp]['id'].get_planning_training_logs()
            ax.plot(d1['steps'], d1['plan_grad'], color=dict[exp]['color'], label=dict[exp]['id'].id)
            ax2.plot(d1['steps'], d1['plan_loss'], color=dict[exp]['color'], label=dict[exp]['id'].id)
    ax.set_title('plan_grad')
    ax2.set_title('plan_loss')
    ax.set_ylabel('plan_grad')
    ax2.set_ylabel('plan_loss')
    ax.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax.grid(True)
    ax2.grid(True)
    ax.set_xlim(0, dis_len)
    ax2.set_xlim(0, dis_len)
    # ax.set_ylim(0, ymax)
    # ax2.set_ylim(0, ymax)
    plt.show()

# ax3 = fig.add_subplot(212)
# f, (ax) = plt.subplots(1, 1, figsize=(16, 8), sharex=True)
    # hide the spines between ax and ax2
    # ax.spines['bottom'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax.xaxis.tick_top()
    # ax.tick_params(labeltop=False)  # don't put tick labels at the top
    # ax2.xaxis.tick_bottom()
    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    # ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal