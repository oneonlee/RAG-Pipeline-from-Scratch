from RAG_pipeline.retrievers.base_retrieval import BaseRetriever
import torch
import textwrap

# Define helper function to print wrapped text 
def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

class DenseRetriever(BaseRetriever):
    def __init__(self, 
                 embedding_model_name: str,
                 library_name: str = "FlagEmbedding",
                ):
        """
        Parameters:
            embedding_model_name (str): A model name to use for embedding the sentence
            library_name (str): Preferred library name when embedding (for developers)
                                Check for the `support_libraries_list`
        """

        support_libraries_list = ["FlagEmbedding", "Sentence-Transformers", "HuggingFace"]
        assert library_name in support_libraries_list, f'"{library_name}" is not supported. Supported libraries: {support_libraries_list}'
            
        super(DenseRetriever, self).__init__()
        self.library_name = library_name
        self.embedding_model, self.tokenizer = self._get_embedding_model(embedding_model_name, library_name)

    
    def _get_embedding_model(self, embedding_model_name, library_name) -> tuple:
        """
        Returns:
            embedding_model (FlagModel or SentenceTransformer or AutoModel): A model for embedding.
                                                                             The type of returned_model depends on `library_name`.
            tokenizer (AutoTokenizer or None): AutoTokenizer if library_name == "HuggingFace", else None.
        """

        tokenizer = None
        if library_name == "FlagEmbedding":
            from FlagEmbedding import FlagModel
            embedding_model = FlagModel(embedding_model_name, 
                              query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                              use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation

        elif library_name == "Sentence-Transformers":
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(model_name_or_path=embedding_model_name)
            
        elif library_name == "HuggingFace":
            from transformers import AutoTokenizer, AutoModel
            # Load model from HuggingFace Hub
            tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
            embedding_model = AutoModel.from_pretrained(embedding_model_name)

        return embedding_model, tokenizer
        

    def encode(self,
               sentences: list[str] or str,
               batch_size: int or None = 256,
               convert_to_tensor: bool = False,
               return_dict: bool = False,
               is_passage: bool = True,
               ) -> dict or np.array or torch.tensor:
        """
        Parameters:
            sentences (list[str] or str): A list of strings each containing the sentence,
                                          or just a single string sentence
            embedding_model_name (str): A model name to use for embedding the sentence
            batch_size (int or None): The batch size used for the computation
            convert_to_tensor (bool): Whether the embedding result should be one large tensor
            return_dict (bool): Whether the output should be zipped dict or just embedding result
            library_name (str): Preferred library name when embedding (for developers)
                                Check for the `support_libraries_list`
            is_passage (bool): Whether the input sentence is passage or query (only meaningful at library_name == "FlagEmbedding")
                                
        Returns:
            embeddings_dict(dict) or embeddings(numpy.array or torch.tensor) : 
                A dictionary, each containing the sentence (string) as a key 
                    & embedding results (numpy.array or torch.tensor) as a value
                or just a embedding result (numpy.array or torch.tensor)
        """

        if isinstance(sentences, str):
            sentences = [sentences]
        
        if self.library_name == "FlagEmbedding":
            if is_passage:
                # Passages are encoded/embedded by calling model.encode()
                if convert_to_tensor:
                    embeddings = self.embedding_model.encode(sentences, batch_size=batch_size, convert_to_numpy=False).view(-1)
                else:
                    embeddings = self.embedding_model.encode(sentences, batch_size=batch_size, convert_to_numpy=True)
            else:
                # Query is encoded/embedded by calling model.encode_queries(), since it doesn't need instruction
                if convert_to_tensor:
                    embeddings = self.embedding_model.encode_queries(sentences, batch_size=batch_size, convert_to_numpy=False).view(-1)
                else:
                    embeddings = self.embedding_model.encode_queries(sentences, batch_size=batch_size, convert_to_numpy=True)
                
        elif self.library_name == "Sentence-Transformers":
            # Sentences are encoded/embedded by calling model.encode()
            embeddings = self.embedding_model.encode(sentences, batch_size=batch_size, convert_to_tensor=convert_to_tensor)
        
        elif self.library_name == "HuggingFace":
            assert batch_size == None, f'"{library_name} mode does not support `batch_size`. Only available when `batch_size` is `None`.'
        
            self.embedding_model.eval()
            
            # Tokenize sentences
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
            # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
        
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.embedding_model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                embeddings = model_output[0][:, 0]
        
            # normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
            if convert_to_tensor:
                pass
            else:
                embeddings = embeddings.numpy()

        if return_dict:
            embeddings_dict = dict(zip(sentences, embeddings))
            return embeddings_dict
        else:
            return embeddings


    def retrieve_relevant_resources(self,
                                    query: str,
                                    embeddings: torch.tensor,
                                    n_resources_to_return: int=5):
        """
        Embeds a query with model and returns top k scores and indices from embeddings.
        """
    
        # Embed the query
        query_embedding = self.encode(query,
                                      convert_to_tensor=True,
                                      is_passage=False)
    
        # Get dot product scores on embeddings
        dot_scores = query_embedding @ embeddings.T
        
        scores, indices = torch.topk(input=dot_scores, 
                                     k=n_resources_to_return)
    
        return scores, indices

    
    def print_top_results_and_scores(self,
                                     query: str,
                                     embeddings: torch.tensor,
                                     pages_and_chunks: list[dict],
                                     n_resources_to_return: int=5):
        """
        Takes a query, retrieves most relevant resources and prints them out in descending order.
    
        Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
        """
        
        scores, indices = self.retrieve_relevant_resources(query=query,
                                                           embeddings=embeddings,
                                                           n_resources_to_return=n_resources_to_return)
        
        print(f"Query: {query}\n")
        print("Results:")
        
        # Loop through zipped together scores and indicies
        for score, index in zip(scores, indices):
            print(f"Score: {score:.4f}")
            # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
            print_wrapped(pages_and_chunks[index]["sentence_chunk"])
            # Print the page number too so we can reference the textbook further and check the results
            print(f"Page number: {pages_and_chunks[index]['page_number']}")
            print("\n")

    
    def retrieve(self, sequence_input_ids, dataset, k=1):
        queries = [
            self._get_query_string(
                sequence_input_ids,
                d["begin_location"],
                d["end_location"],
                d["title"] if "title" in d else None
            )
            for d in dataset
        ]
        assert len(queries) == len(dataset)
        all_res = self.searcher.batch_search(
            queries,
            qids=[str(i) for i in range(len(queries))],
            k=max(100, 4*k) if self.forbidden_titles else k,
            threads=multiprocessing.cpu_count()
        )

        for qid, res in all_res.items():
            qid = int(qid)
            d = dataset[qid]
            d["query"] = queries[qid]
            allowed_docs = []
            for hit in res:
                res_dict = json.loads(hit.raw)
                context_str = res_dict["contents"]
                title = self._get_title_from_retrieved_document(context_str)
                
                allowed_docs.append({"text": context_str, "score": hit.score})
                if len(allowed_docs) >= k:
                    break
            d["retrieved_docs"] = allowed_docs
        return dataset