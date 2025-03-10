import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix


class SimilarityGraph:
    """
    Construct Similarity graph and implement HMRF in spatial transcriptomics
    """

    def __init__(
        self,
        adata,
        kneighbors,
        beta,
        alpha,
        theta,
        init_alpha,
        icm_iter: int = 2,
        max_iter: int = 3,
        convergency_threshold: float = 1e-5,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self.matrix = adata.to_df()
        self.cell_num = adata.shape[0]
        self.kneighbors = kneighbors + 1
        self.graph = self._construct_graph(adata.obsm["spatial"], self.kneighbors)
        self.beta = beta
        self.icm_iter = icm_iter
        self.max_iter = max_iter
        self.alpha = alpha
        self.theta = theta
        self.init_alpha = init_alpha
        self.convergency_threshold = convergency_threshold
        self.neighbor_corr = self._neighbor_init(self.alpha)
        if self.verbose:
            print(
                f"Initialized model with beta: {self.beta}, alpha: {alpha}, theta: {self.theta}"
            )

    def fit(
        self,
        gene_id: str,
    ) -> None:
        """
        Implement HMRF using ICM-EM
        """
        self.exp = self.matrix[gene_id].values
        self._initialize_labels(self.init_alpha)
        self._update_adj_matrix(self.theta)
        self._run_icmem(
            self.beta,
            self.theta,
            self.icm_iter,
            self.max_iter,
            convergency_threshold=self.convergency_threshold,
        )

    def _initialize_labels(self, init_alpha: float) -> None:
        """
        Initialize label with smoothed expression matrix
        """
        neighbor_corr = self.neighbor_corr.copy()
        neighbor_corr = neighbor_corr / self.alpha * init_alpha
        neighbor_corr.setdiag(1)
        smoothed_exp = neighbor_corr.dot(self.exp)
        gmm = GaussianMixture(n_components=2).fit(smoothed_exp.reshape(-1, 1))
        means, covs = gmm.means_.ravel(), gmm.covariances_.ravel()  # type: ignore
        self.cls_para = np.column_stack((means, covs))
        self.labels = gmm.predict(smoothed_exp.reshape(-1, 1))
        self._label_resort()

    def _impute(self) -> csr_matrix:
        """
        Impute the expression by considering neighbor cells
        """
        return self.adj_matrix.dot(self.exp)

    def _construct_graph(self, coord: np.ndarray, kneighbors: int = 18) -> csr_matrix:
        """
        Construct gene graph based on the nearest neighbors
        """
        if self.verbose:
            print("Constructing Graph")
        graph = (
            NearestNeighbors(n_neighbors=kneighbors).fit(coord).kneighbors_graph(coord)
        )
        self.cell_neighbors = graph.indices.reshape(self.cell_num, kneighbors)
        return graph

    def _label_resort(self) -> None:
        """
        Set the label with the highest mean as 1
        """
        means = self.cls_para[:, 0]
        cls_label = np.argmax(means)
        new_labels = np.zeros_like(self.labels)
        new_labels[self.labels == cls_label] = 1
        self.labels = new_labels

    def _run_icmem(
        self,
        beta,
        theta,
        icm_iter: int = 2,
        max_iter: int = 3,
        convergency_threshold: float = 1e-5,
    ) -> None:
        """
        Run ICM-EM algorithm to update gene panel's labels and integrate neighbor spots expression
        """
        sqrt2pi = np.sqrt(2 * np.pi)
        cell_num = self.cell_num
        temp = 1  # TODO add melting mechanism
        iteration = 0
        converged = False
        while iteration < max_iter and not converged:
            # ICM step
            changed = 0
            for _ in range(icm_iter):
                indices = np.arange(cell_num)
                new_labels = 1 - self.labels[indices]
                delta_energies = self._delta_energies(indices, new_labels, beta)
                negative_indices = delta_energies < 0
                self.labels[indices[negative_indices]] = new_labels[negative_indices]
                changed += np.sum(negative_indices)

                # Metropolis-Hastings
                non_negative_indices = np.logical_not(negative_indices)
                probabilities = np.exp(-delta_energies[non_negative_indices] / temp)
                probabilities[probabilities == 0] = 1e-5
                samples = np.random.uniform(0, 1, size=probabilities.shape)
                update = samples < probabilities
                self.labels[indices[non_negative_indices][update]] = new_labels[
                    non_negative_indices
                ][update]
                changed += np.sum(update)

                if changed == 0:
                    break

            # EM step initialization
            means, vars = self.cls_para.T
            vars[np.isclose(vars, 0)] = 1e-5
            squared_diff = (self.exp[:, None] - means) ** 2

            # E step
            clusterProb = np.exp(-0.5 * squared_diff / vars) / (sqrt2pi * np.sqrt(vars))
            clusterProb[np.isclose(clusterProb, 0)] = 1e-5
            clusterProb = clusterProb / clusterProb.sum(axis=1)[:, None]

            # M Step
            weights = clusterProb / clusterProb.sum(axis=0)
            means = np.sum(self.exp[:, None] * weights, axis=0)
            vars = np.sum(weights * squared_diff, axis=0) / weights.sum(axis=0)
            vars[np.isclose(vars, 0)] = 1e-5

            new_para = np.column_stack([means, vars])
            para_change = np.max(np.abs(new_para - self.cls_para))
            if para_change < convergency_threshold:
                converged = True
            self.cls_para = new_para
            # Update expression matrix
            if changed > 0:
                self._update_adj_matrix(theta)
            self.exp = self._impute()
            iteration += 1
        return

    def _delta_energies(self, indices, new_labels, beta) -> np.ndarray:
        neighbor_indices = self.cell_neighbors
        means, vars = self.cls_para[1 - new_labels].T
        new_means, new_vars = self.cls_para[new_labels].T
        sqrt_2_pi_vars = np.sqrt(2 * np.pi * vars)
        sqrt_2_pi_new_vars = np.sqrt(2 * np.pi * new_vars)

        delta_energy_consts = (
            np.log(sqrt_2_pi_new_vars / sqrt_2_pi_vars)
            + ((self.exp[indices] - new_means) ** 2 / (2 * new_vars))
            - ((self.exp[indices] - means) ** 2 / (2 * vars))
        )

        delta_energy_neighbors = (
            beta
            * 2
            * np.sum(
                self._difference(new_labels, self.labels[neighbor_indices])
                - self._difference(self.labels[indices], self.labels[neighbor_indices]),
                axis=0,
            )
            / self.kneighbors
        )

        return delta_energy_consts + delta_energy_neighbors

    def _neighbor_init(self, alpha, n_comp=15) -> csr_matrix:
        """
        Initialize the neighboring correlation matrix
        """

        if self.verbose:
            print("Initializing neighbor correlation matrix")
        pca = PCA(n_comp).fit_transform(StandardScaler().fit_transform(self.matrix))
        pca_centered = pca - np.mean(pca, axis=1, keepdims=True)
        norms = np.linalg.norm(pca_centered, axis=1)
        norms[norms == 0] = 1e-5
        pca_normalized = pca_centered / norms[:, np.newaxis]
        # Get graph edges
        graph_coo = self.graph.tocoo()
        row_indices = graph_coo.row
        col_indices = graph_coo.col
        correlations = np.exp(
            (pca_normalized[row_indices] * pca_normalized[col_indices]).sum(axis=1)
        )
        neighbor_corr = csr_matrix(
            (correlations, (row_indices, col_indices)),
            shape=(self.cell_num, self.cell_num),
        )
        neighbor_corr.setdiag(0)
        neighbor_corr = self._csr_normalize(neighbor_corr)
        neighbor_corr = neighbor_corr.multiply(alpha)
        neighbor_corr.setdiag(1)
        neighbor_corr = self._csr_normalize(neighbor_corr)

        return neighbor_corr

    def _update_adj_matrix(self, theta: float) -> None:
        """
        Efficiently update the adjacency matrix based on the labels.

        Parameters:
        ----------
        theta : float
            Scaling factor for off-diagonal entries where labels do not match.
        """

        neighbor_corr_coo = self.neighbor_corr.tocoo()
        row_indices = neighbor_corr_coo.row
        col_indices = neighbor_corr_coo.col
        data = neighbor_corr_coo.data

        diff_label = self.labels[row_indices] != self.labels[col_indices]
        new_data = data.copy()
        new_data[diff_label] *= theta

        self.adj_matrix = self._csr_normalize(
            csr_matrix(
                (new_data, (row_indices, col_indices)), shape=self.neighbor_corr.shape
            )
        )

    @staticmethod
    def _csr_normalize(mat) -> csr_matrix:
        """
        Normalize the csr matrix
        """
        row_sums = np.array(mat.sum(axis=1)).flatten()
        inv_rs = 1.0 / row_sums
        mat = mat.multiply(inv_rs[:, np.newaxis])
        return mat

    @staticmethod
    def _difference(x, y):
        return np.abs(np.subtract(x, y.T))
