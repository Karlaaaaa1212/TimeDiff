
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from torch import optim
import pickle


def _join_with_history(ind_out, history, series):
    """
    Prepend the last history point so that the forecast line starts exactly where
    the black history line ends (no visual gap between history and prediction).
    """
    x = [ind_out[0] - 1] + list(ind_out)
    y = [np.asarray(history).reshape(-1)[-1]] + list(np.asarray(series).reshape(-1))
    return x, y


def visual(args, history, true, preds=None, mean_pred=None, label_part=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(8,5))
    ind_his = list(np.arange(0,len(history)))
    ind_out = list(np.arange(len(history), len(history)+len(true)))
    if label_part is not None:
        label_out = list(np.arange(len(history)-len(label_part), len(history)))
    plt.plot(ind_his, history, '-', label='History', c='#000000', linewidth=1)

    x, y = _join_with_history(ind_out, history, true)
    plt.plot(x, y, '-', label='GroundTruth', c='b', linewidth=1) # #999999
    if mean_pred is not None:
        x, y = _join_with_history(ind_out, history, mean_pred)
        plt.plot(x, y, '-', label='Pred-Trend', c='gray', linewidth=1)
    if preds is not None:
        # print(np.shape(ind_out), np.shape(preds))
        x, y = _join_with_history(ind_out, history, preds)
        plt.plot(x, y, '-', label='Prediction', c='r', linewidth=1)  # #FFB733
    if label_part is not None:
        plt.plot(label_out, label_part, '-', label='Pred-Label', c='pink', linewidth=1)

    plt.axvline(x=ind_out[0] - 0.5, color='#999999', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')

    f = open(name[:-4]+'.pkl', "wb")
    pickle.dump(preds, f)
    f.close()

    f = open(name[:-4]+'_ground_truth.pkl', "wb")
    pickle.dump(true, f)
    f.close()

    f = open(name[:-4]+'_history.pkl', "wb")
    pickle.dump(history, f)
    f.close()

def visual_prob(args, history, true, preds=None, mean_pred=None, label_part=None, name='./pic/test.pdf', prob_pd=None):
    """
    Results visualization
    """
    plt.figure(figsize=(8,5))
    ind_his = list(np.arange(0,len(history)))
    ind_out = list(np.arange(len(history), len(history)+len(true)))
    if label_part is not None:
        label_out = list(np.arange(len(history)-len(label_part), len(history)))
    plt.plot(ind_his, history, '-', label='History', c='#000000', linewidth=1)

    x, y = _join_with_history(ind_out, history, true)
    plt.plot(x, y, '-', label='GroundTruth', c='b', linewidth=1) # #999999
    if mean_pred is not None:
        x, y = _join_with_history(ind_out, history, mean_pred)
        plt.plot(x, y, '-', label='Pred-Trend', c='gray', linewidth=1)
    if preds is not None:
        mean = np.mean(preds, axis=0).reshape(-1, 1)
        std = np.std(preds, axis=0).reshape(-1, 1)

        if args.sample_times > 1:
            ub = mean + std
            lb = mean - std
            new_ind_out = np.expand_dims(np.array(ind_out), axis=1)[:,0]
            plt.fill_between(new_ind_out, ub[:,0], lb[:,0], color="#b9cfe7", edgecolor=None)
        # plt.fill_between(ind_out, mean + std, mean - std, facecolor="gray")
        x, y = _join_with_history(ind_out, history, mean)
        plt.plot(x, y, '-', label='Prediction', c='r', linewidth=1)  # #FFB733
    if label_part is not None:
        plt.plot(label_out, label_part, '-', label='Pred-Label', c='pink', linewidth=1)

    plt.axvline(x=ind_out[0] - 0.5, color='#999999', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    print(name)
    # plt.show()
    plt.savefig(name, bbox_inches='tight') 
    
    f = open(name[:-4]+'.pkl', "wb")
    pickle.dump(preds, f)
    f.close()

    f = open(name[:-4]+'_ground_truth.pkl', "wb")
    pickle.dump(true, f)
    f.close()

    f = open(name[:-4]+'_history.pkl', "wb")
    pickle.dump(history, f)
    f.close()


def visual2D(history, true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    # print(np.shape(history), np.shape(true), np.shape(preds))

    gtrue = np.concatenate([history, true], axis=0)
    preds = np.concatenate([history, preds], axis=0)

    # print(np.shape(gtrue), np.shape(preds))

    plt.figure(figsize=(14,5))
    cmap = 'jet'
    aspect = 1
    plt.subplot(211)
    plt.imshow(gtrue.T, cmap=cmap, aspect=aspect, vmin=-0.3, vmax=0.3)
    plt.subplot(212)
    plt.imshow(preds.T, cmap=cmap, aspect=aspect, vmin=-0.3, vmax=0.3)
    plt.tight_layout()
    plt.savefig(name)

    


def plot_loss_curve(training_process, name='./loss_curve.png', title=None):
    """
    Draw the training curve so you can tell whether the loss actually goes down.

    training_process: {"train_loss": [...], "val_loss": [...]}
        train_loss -> diffusion objective on the train set (per epoch)
        val_loss   -> forecasting MSE on the validation set (per epoch)
    """
    train_loss = list(training_process.get("train_loss", []))
    val_loss = list(training_process.get("val_loss", []))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if len(train_loss) > 0:
        ep = np.arange(1, len(train_loss) + 1)
        axes[0].plot(ep, train_loss, '-o', ms=2.5, c='#1f77b4')
        best = int(np.argmin(train_loss))
        axes[0].scatter([best + 1], [train_loss[best]], c='r', zorder=5,
                        label='min @ epoch {} = {:.6f}'.format(best + 1, train_loss[best]))
        axes[0].legend()
    axes[0].set_title('train loss (diffusion objective)')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss'); axes[0].grid(alpha=0.3)

    if len(val_loss) > 0:
        ep = np.arange(1, len(val_loss) + 1)
        axes[1].plot(ep, val_loss, '-o', ms=2.5, c='#d62728')
        best = int(np.argmin(val_loss))
        axes[1].scatter([best + 1], [val_loss[best]], c='k', zorder=5,
                        label='min @ epoch {} = {:.6f}'.format(best + 1, val_loss[best]))
        axes[1].legend()
    axes[1].set_title('val loss (forecast MSE, checkpoint criterion)')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('mse'); axes[1].grid(alpha=0.3)

    if title is not None:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight', dpi=120)
    plt.close(fig)

    def _trend(v):
        if len(v) < 2:
            return "n/a"
        k = max(1, len(v) // 10)
        return "first{}avg={:.6f} -> last{}avg={:.6f} ({:+.1f}%)".format(
            k, float(np.mean(v[:k])), k, float(np.mean(v[-k:])),
            100.0 * (np.mean(v[-k:]) - np.mean(v[:k])) / (abs(np.mean(v[:k])) + 1e-12))

    print("[loss curve] saved to", name)
    print("[loss curve] train:", _trend(train_loss))
    print("[loss curve] val  :", _trend(val_loss))
