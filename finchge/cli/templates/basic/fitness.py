from finchge.fitness import GEFitnessFunction


class StringMatchFitness(GEFitnessFunction):
    def __init__(self, target):
        super().__init__(maximize=False)
        self.target = target

    def evaluate(self, context):
        phenotype = context["phenotype"]

        max_len = max(len(self.target), len(phenotype))
        min_len = min(len(self.target), len(phenotype))

        matches = sum(
            t == g for t, g in zip(self.target[:min_len], phenotype[:min_len])
        )

        return max_len - matches
