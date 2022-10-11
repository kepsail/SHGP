from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def visual_embd(embd, label, seed=0):
    visual_vec = TSNE(n_components=2).fit_transform(embd)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(visual_vec[:,0], visual_vec[:,1], c=label, s=3)

    plt.show()

    fig_name = 'test.jpg'
    plt.savefig(fig_name)
    plt.close(fig)
