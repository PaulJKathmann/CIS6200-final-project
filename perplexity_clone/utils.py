import json
import os
from pydantic import BaseModel

class BiasScore(BaseModel):
    score: float
    probability_left: float
    probability_center: float
    probability_right: float

class BiasLog(BaseModel):
    source_bias_scores: list[BiasScore]
    total_source_bias_score: float | None
    output_bias_score: BiasScore | None
    output: str | None
    source: str | None

class BiasLogger:
    def __init__(self, log_file_path: str = None, verbose: bool = False):
        self.log_file_path = log_file_path or 'experiment/experiment_results/bias_scores_4.json'
        self.bias_scores_data = self.load_bias_scores()
        self.verbose = verbose
    
    def load_bias_scores(self) -> dict[str, BiasLog]:
        if os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'r') as file:
                file = json.load(file)
                return {k: BiasLog.model_validate(v) for k, v in file.items()}
        return {}
    
    def log_source_bias_score(self, prompt: str, bias_probabilities: list[float], bias_score: float) -> None:
        print(f"Logging source bias score for prompt '{prompt}': {bias_probabilities}, {bias_score}")
        if prompt not in self.bias_scores_data:
            self.bias_scores_data[prompt] = BiasLog(source_bias_scores=[], total_source_bias_score=None , output_bias_score=None, output=None, source=None)
        bias_score = BiasScore(score=bias_score, 
                               probability_left=bias_probabilities[0], 
                               probability_center=bias_probabilities[1], 
                               probability_right=bias_probabilities[2]
                               )
        self.bias_scores_data[prompt].source_bias_scores.append(bias_score)
        if self.verbose:
            print(f"Logged source bias score for prompt '{prompt}': {bias_score}, ")
    
    def log_total_source_bias_score(self, prompt: str, total_bias_score: float) -> None:
        if prompt not in self.bias_scores_data:
            raise ValueError(f"Source bias scores for prompt '{prompt}' have not been logged yet. Please log source bias scores first.")
        self.bias_scores_data[prompt].total_source_bias_score = total_bias_score
        if self.verbose:
            print(f"Logged total source bias score for prompt '{prompt}': {total_bias_score}, ")

    def log_output_bias_score(self, prompt: str, bias_probabilities: list[float], bias_score: float, output: str) -> None:
        if prompt not in self.bias_scores_data:
            raise ValueError(f"Source bias scores for prompt '{prompt}' have not been logged yet. Please log source bias scores first.")
        self.bias_scores_data[prompt].output_bias_score = BiasScore(score=bias_score, 
                                                                    probability_left=bias_probabilities[0], 
                                                                    probability_center=bias_probabilities[1], 
                                                                    probability_right=bias_probabilities[2]
                                                                    )
        self.bias_scores_data[prompt].output = output
        if self.verbose:
            print(f"Logged output bias score for prompt '{prompt}': {bias_score}, ")
        self.save()
        
    def save(self) -> None:
        with open(self.log_file_path, 'w') as file:
            json.dump({k: v.model_dump()  for k,v in self.bias_scores_data.items()}, file, indent=4)
    


            
       


    