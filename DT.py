class DecisionTree:
    def __init__(self, training_data, header):
        """
        Inisialisasi Decision Tree dengan data training dan header kolom.
        :param training_data: List of data rows (2D list).
        :param header: List of column names.
        """
        self.training_data = training_data
        self.header = header
        self.tree = self.build_tree(self.training_data)

    @staticmethod
    def unique_vals(rows, col):
        """Mengembalikan nilai unik dalam sebuah kolom."""
        return set([row[col] for row in rows])

    @staticmethod
    def class_counts(rows):
        """Menghitung jumlah tiap kelas dalam dataset."""
        counts = {}
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    @staticmethod
    def is_numeric(value):
        """Memeriksa apakah sebuah nilai adalah numerik."""
        return isinstance(value, int) or isinstance(value, float)

    class Question:
        def __init__(self, column, value):
            self.column = column
            self.value = value

        def match(self, example):
            val = example[self.column]
            if DecisionTree.is_numeric(val):
                return val >= self.value
            else:
                return val == self.value

        def __repr__(self):
            condition = "=="
            if DecisionTree.is_numeric(self.value):
                condition = ">="
            return f"Is column[{self.column}] {condition} {str(self.value)}?"

    @staticmethod
    def partition(rows, question):
        """Memisahkan dataset berdasarkan pertanyaan."""
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    @staticmethod
    def gini(rows):
        """Menghitung impurity Gini."""
        counts = DecisionTree.class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity

    @staticmethod
    def info_gain(left, right, current_uncertainty):
        """Menghitung informasi gain."""
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * DecisionTree.gini(left) - (1 - p) * DecisionTree.gini(right)

    def find_best_split(self, rows):
        """Mencari split terbaik untuk dataset."""
        best_gain = 0
        best_question = None
        current_uncertainty = self.gini(rows)
        n_features = len(rows[0]) - 1

        for col in range(n_features):
            values = set([row[col] for row in rows])
            for val in values:
                question = self.Question(col, val)
                true_rows, false_rows = self.partition(rows, question)

                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                gain = self.info_gain(true_rows, false_rows, current_uncertainty)

                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    class Leaf:
        def __init__(self, rows):
            self.predictions = DecisionTree.class_counts(rows)

    class Decision_Node:
        def __init__(self, question, true_branch, false_branch):
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch

    def build_tree(self, rows):
        """Membangun pohon keputusan."""
        gain, question = self.find_best_split(rows)
        if gain == 0:
            return self.Leaf(rows)

        true_rows, false_rows = self.partition(rows, question)
        true_branch = self.build_tree(true_rows)
        false_branch = self.build_tree(false_rows)

        return self.Decision_Node(question, true_branch, false_branch)

    def classify(self, row, node=None):
        """Mengklasifikasi sebuah instance."""
        if node is None:
            node = self.tree

        if isinstance(node, self.Leaf):
            return node.predictions

        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)

    @staticmethod
    def print_leaf(counts):
        """Format hasil prediksi."""
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs

    def print_tree(self, node=None, spacing=""):
        """Mencetak representasi pohon keputusan."""
        if node is None:
            node = self.tree

        if isinstance(node, self.Leaf):
            print(spacing + "Predict", node.predictions)
            return

        print(spacing + str(node.question))
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")
        print(spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")
