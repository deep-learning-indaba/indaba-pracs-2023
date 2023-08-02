import torch
import jraph
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
from torch_geometric.data import Data


class UniGraphDataPreparation:
    def __init__(
        self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, embedding_dim: int = 19
    ):
        """
        Initialize the class with ratings and movies dataframes.

        Args:
            ratings_df: DataFrame with ratings data.
            movies_df: DataFrame with movies data.
            embedding_dim: The desired dimension of the user embeddings.
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.embedding_dim = embedding_dim

        # Create mappings for userId and movieId to new continuous indexes
        self.user_mapping = {
            user_id: i for i, user_id in enumerate(self.ratings_df.userId.unique())
        }
        self.movie_mapping = {
            movie_id: i for i, movie_id in enumerate(self.ratings_df.movieId.unique())
        }

        # Apply the mappings to set new continuous userIds and movieIds
        self.ratings_df["userId"] = self.ratings_df["userId"].map(self.user_mapping)
        self.ratings_df["movieId"] = self.ratings_df["movieId"].map(self.movie_mapping)
        self.movies_df["movieId"] = self.movies_df["movieId"].map(self.movie_mapping)

    def create_edge_index(self):
        """
        Create an edge index for the graph. Edge direction is from user to movie.

        Returns:
            A tensor representing the edge index.
        """
        user_nodes = self.ratings_df["userId"].to_numpy()
        movie_nodes = (
            self.ratings_df["movieId"].to_numpy() + self.ratings_df["userId"].nunique()
        )

        edge_index = torch.tensor(np.array([user_nodes, movie_nodes]), dtype=torch.long)

        return edge_index

    def create_edge_features(self):
        """
        Create edge features for the graph.

        Returns:
            A tensor representing the edge features.
        """
        ratings = self.ratings_df["rating"].to_numpy()
        edge_attr = torch.tensor(ratings, dtype=torch.float).view(-1, 1)

        return edge_attr

    def create_node_features(self):
        """
        Create node features for the graph.

        Returns:
            A tensor representing the node features.
        """
        # Prepare user features
        num_users = self.ratings_df["userId"].nunique()
        user_embeddings = nn.Embedding(num_users, self.embedding_dim)
        user_features = user_embeddings.weight.detach().numpy()

        # Prepare movie features
        movie_genres = self.movies_df["genres"].str.split("|")
        mlb = MultiLabelBinarizer()
        movie_features = mlb.fit_transform(movie_genres)

        # Combine user and movie features
        node_features = np.vstack([user_features, movie_features])
        node_features = torch.tensor(node_features, dtype=torch.float)

        return node_features

    def split_graph(self, node_features, edge_attr, edge_index, train_ratio=0.8):
        # Randomly permute the edge indices
        perm = torch.randperm(edge_index.shape[1])

        # Split the edge indices into train and test
        train_edge_index = edge_index[:, perm[: int(train_ratio * perm.size(0))]]
        test_edge_index = edge_index[:, perm[int(train_ratio * perm.size(0)) :]]

        # Split the edge features into train and test
        train_edge_attr = edge_attr[perm[: int(train_ratio * perm.size(0))]]
        test_edge_attr = edge_attr[perm[int(train_ratio * perm.size(0)) :]]

        # Create Data objects for train and test sets
        train_graph = Data(
            x=node_features, edge_index=train_edge_index, edge_attr=train_edge_attr
        )
        test_graph = Data(
            x=node_features, edge_index=test_edge_index, edge_attr=test_edge_attr
        )

        return train_graph, test_graph

    def prepare_data(self):
        """
        Prepare the graph data.

        Returns:
            A train and test PyG Data object with the prepared graph data.
        """
        edge_index = self.create_edge_index()
        edge_attr = self.create_edge_features()
        x = self.create_node_features()

        train_graph, test_graph = self.split_graph(
            x, edge_attr, edge_index, train_ratio=0.8
        )

        train_jraph = jraph.GraphsTuple(
            nodes=np.array(train_graph.x),
            edges=np.array(train_graph.edge_attr),
            n_node=np.array([train_graph.num_nodes]),
            n_edge=np.array([train_graph.num_edges]),
            senders=np.array(train_graph.edge_index[0]),  # users are source
            receivers=np.array(train_graph.edge_index[1]),  # movies are recievers
            globals=np.array([]),
        )
        test_jraph = jraph.GraphsTuple(
            nodes=np.array(test_graph.x),
            edges=np.array(test_graph.edge_attr),
            n_node=np.array([test_graph.num_nodes]),
            n_edge=np.array([test_graph.num_edges]),
            senders=np.array(test_graph.edge_index[0]),  # users are source
            receivers=np.array(test_graph.edge_index[1]),  # movies are recievers
            globals=np.array([]),
        )

        return train_jraph, test_jraph


def subset_dataset(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    top_users: int = 10000,
    top_movies: int = 1000,
):
    """
    Get subset of the data, particularly only pick top N users and M movies.

    Args:
        ratings_df: DataFrame with ratings data.
        movies_df: DataFrame with movies data.
        top_users: The number of top users to include in the graph.
        top_movies: The number of top movies to include in the graph.
        embedding_dim: The desired dimension of the user embeddings.

    Returns:
        A subset of the rating and movies dataframes.
    """
    # Select top users and movies
    top_users_ids = ratings_df.userId.value_counts().index[:top_users]
    top_movies_ids = ratings_df.movieId.value_counts().index[:top_movies]

    # Filter the dataframes
    ratings_subset_df = ratings_df[
        ratings_df.userId.isin(top_users_ids) & ratings_df.movieId.isin(top_movies_ids)
    ].copy()
    movies_subset_df = movies_df[movies_df.movieId.isin(top_movies_ids)].copy()

    # Convert timestamp to readable date
    ratings_subset_df["date"] = pd.to_datetime(ratings_subset_df["timestamp"], unit="s")

    return ratings_subset_df, movies_subset_df
