import numpy as np
from anndata import AnnData
from joblib import Parallel, delayed
from tqdm import tqdm
from .graph import SimilarityGraph
from .utils import feature_ranking, lap_score


class CadaST:
    """
    Construct gene graph and implement HMRF in spatial transcriptomics
    Accerlate the process by using joblib multi-process
    """

    def __init__(
        self,
        adata: AnnData,
        kneighbors: int,
        beta: int = 10,
        alpha: float = 0.6,
        theta: float = 0.2,
        init_alpha: float = 6,
        icm_iter: int = 1,
        max_iter: int = 3,
        n_top: int | None = None,
        n_jobs: int = 16,
        verbose: bool = True,
    ):
        self.adata = adata.copy()
        self.kneighbors = kneighbors
        self.beta = beta
        self.alpha = alpha
        self.theta = theta
        self.init_alpha = init_alpha
        self.max_iter = max_iter
        self.icm_iter = icm_iter
        self.gene_list = self.adata.var_names
        self.n_top = n_top
        self.n_jobs = n_jobs
        self.graph = None
        self.verbose = verbose

    def construct_graph(self) -> None:
        """
        Construct gene graph
        """
        graph = SimilarityGraph(
            adata=self.adata,
            kneighbors=self.kneighbors,
            beta=self.beta,
            alpha=self.alpha,
            theta=self.theta,
            init_alpha=self.init_alpha,
            icm_iter=self.icm_iter,
            max_iter=self.max_iter,
            verbose=self.verbose,
        )
        self.graph = graph

    def filter_genes(self, n_top=2000) -> None:
        """
        Filter genes with top SVG features
        """
        if self.graph is None:
            self.construct_graph()
        n_top = n_top if self.n_top is None else self.n_top
        if (n_top is not None) and (n_top < len(self.gene_list)):
            if self.verbose:
                print(f"Filtering genes with top {n_top} SVG features")
            lapScore = lap_score(self.adata.X, self.graph.neighbor_corr) # type: ignore
            feature_rank = feature_ranking(lapScore)
            self.gene_list = self.gene_list[feature_rank[:n_top]]
            self.adata = self.adata[:, self.gene_list] # type: ignore

    def fit(self) -> AnnData:
        """
        Fit the cadaST model
        """
        if self.graph is None:
            self.construct_graph()
        if (self.n_top is not None) and (self.n_top < len(self.gene_list)):
            self.filter_genes()
        print("Start cadaST model fitting")
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_gene)(
                self.graph,
                gene,
            )
            for gene in tqdm(self.gene_list)
        )
        imputed_exp, labels = zip(*results)
        self.adata.X = np.array(imputed_exp).T
        self.adata.layers["labels"] = np.array(labels).T
        return self.adata

    @staticmethod
    def _process_gene(model, gene) -> tuple:
        model.fit(
            gene_id=gene,
        )
        return model.exp, model.labels
