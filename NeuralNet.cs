using System;
using System.Collections.Generic;

/*
Cost = (activation - desired output)^2
dCost/dWeight = (activation of prev layer) (deriv of activation func of value before activation) (2) (activation - desired output)
dCost/dBias = (deriv of activation func of value before activation) (2) (activation - desired output)
*/

namespace BigBrian {
    public class NeuralNet {

        public static double TrimMod = 0.01;

        public double Cost { get; set; }
        
        private int[] structure;
        public int[] Structure { get => structure; }

        private Layer[] layers;

        public NeuralNet(int[] structure) {
            this.structure = structure;
        }

        public NeuralNet(int[] structure, int seed) {
            this.structure = structure;

            Gen(seed);
        }

        public NeuralNet(NeuralNet copyNet) { } // TODO

        public void Gen(int seed) {
            layers = new Layer[structure.Length - 1];
            for (int i = 1; i < structure.Length; i++) {
                layers[i - 1] = new Layer(structure[i], structure[i - 1], seed);
            }
        }

        public double[] Passthrough(double[] initLayer) {
            for (int i = 0; i < layers.Length; i++) {
                initLayer = layers[i].Passthrough(initLayer);
            }
            return initLayer;
        }

        public double[] Passthrough(double[] initLayer, double[] target) {
            var output = Passthrough(initLayer);
            CalculateCost(output, target);
            return output;
        }

        public void CalculateCost(double[] output, double[] target) {
            Cost = 0;
            for (int i = 0; i < output.Length; i++) {
                Cost += Math.Pow(output[i] - target[i], 2);
            }
            // Average??? idfk
        }

        public void CalculateDeltas(double[] output, double[] target) {
            for (int i = layers.Length - 1; i >= 0; i--) {
                if (i == layers.Length - 1)
                    layers[i].CalculateDeltas(output, target);
                else
                    layers[i].CalculateDeltas(layers[i+1]);

                layers[i].CacheDeltas();
            }
        }

        public void ApplyDeltas(bool clearCache) {
            foreach (var l in layers) {
                l.ApplyDeltas(clearCache);
            }
        }

        public class Layer {

            private Random random;

            private int size;
            public int Size { get => size; }
            private int weightsPerNode;
            public int WeightsPerNode { get => weightsPerNode; }

            public double bias;
            public double[] nodes;
            public double[][] weights;
            public double biasDelta;
            public double[][] weightDeltas;

            public List<DeltaCache> Deltas = new List<DeltaCache>();
            public TrainingDataDump trainingMeta;

            public Layer(int size, int weightsPerNode) {
                this.size = size;
                this.weightsPerNode = weightsPerNode;
            }

            public Layer(int size, int weightsPerNode, int seed) {
                this.size = size;
                this.weightsPerNode = weightsPerNode;

                Gen(seed);
            }
            
            public Layer(Layer copyLayer) {

            }

            public void Gen(int seed) {

                random = new Random(seed);

                nodes = new double[size];
                weights = new double[weightsPerNode][];
                for (int i = 0; i < weightsPerNode; i++) {
                    weights[i] = new double[size];
                    for (int j = 0; j < size; j++) {
                        weights[i][j] = random.NextDouble() - 0.5; // -0.5 >= x < 0.5
                    }
                }
                bias = (random.NextDouble() * 1.0) - 0.5; // -0.5 >= x < 0.5
            }

            public double[] Passthrough(double[] previousNodes) {
                trainingMeta.FedData = previousNodes;
                trainingMeta.BeforeActivation = new double[size];
                // trainingMeta.AfterActivation = new double[size];

                for (int i = 0; i < size; i++) {
                    nodes[i] = 0.0;
                    for (int j = 0; j < previousNodes.Length; j++) {
                        nodes[i] += weights[j][i] * previousNodes[j];
                    }
                    nodes[i] += bias;
                    trainingMeta.BeforeActivation[i] = nodes[i];
                    nodes[i] = Activation(nodes[i]);
                    // trainingMeta.AfterActivation[i] = nodes[i];
                }
                return nodes;
            }

            public void CalculateDeltas(double[] output, double[] target) {

                // Weights
                weightDeltas = new double[weights.Length][];
                for (int i = 0; i < weightDeltas.Length; i++) {
                    weightDeltas[i] = new double[size];
                    for (int j = 0; j < output.Length; j++) {
                        weightDeltas[i][j] = (1.0 / (double)output.Length) * 2 * (output[j] - target[j]) * ActivationDer(trainingMeta.BeforeActivation[j]) * trainingMeta.FedData[i];
                    }
                }

                // Bias
                for (int j = 0; j < output.Length; j++) {
                    biasDelta += (1.0 / (double)output.Length) * 2 * (output[j] - target[j]) * ActivationDer(trainingMeta.BeforeActivation[j]);
                }
                // biasDelta /= output.Length; // Average out the bias delta cuz idfk what to do
            }

            public void CalculateDeltas(Layer foreLayer) {
                // Weights
                weightDeltas = new double[weights.Length][];
                for (int i = 0; i < weightDeltas.Length; i++) {
                    weightDeltas[i] = new double[size];
                    for (int j = 0; j < size; j++) {
                        weightDeltas[i][j] = 0;
                        for (int k = 0; k < foreLayer.size; k++) {
                            weightDeltas[i][j] += (foreLayer.weightDeltas[j][k] / foreLayer.trainingMeta.FedData[k])
                                * ActivationDer(trainingMeta.BeforeActivation[j]) * trainingMeta.FedData[i];
                        }
                        weightDeltas[i][j] /= foreLayer.size;
                    }
                }

                // Bias
                biasDelta = 0;
                for (int j = 0; j < size; j++) {
                    for (int i = 0; i < weights.Length; i++) {
                        double tempBiasDelta = 0.0;
                        for (int k = 0; k < foreLayer.size; k++) {
                            tempBiasDelta += (foreLayer.weightDeltas[j][k] / foreLayer.trainingMeta.FedData[k])
                                * ActivationDer(trainingMeta.BeforeActivation[j]);
                        }
                        biasDelta += tempBiasDelta;// / foreLayer.size;
                    }
                }
                biasDelta /= size; // Average out the bias delta cuz idfk what to do
            }

            public void CacheDeltas() {
                Deltas.Add(new DeltaCache() { biasDelta = this.biasDelta, weightsDelta = this.weightDeltas });
            }

            public void ApplyDeltas(bool clearCache) {
                // Get empty deltas;
                double[][] dW = weightDeltas;
                for (int x = 0; x < weightDeltas.Length; x++) {
                    for (int y = 0; y < weightDeltas[x].Length; y++) {
                        dW[x][y] = 0.0;
                    }
                }
                double dB = 0.0;

                // Sum
                foreach (var deltaSet in Deltas) {
                    for (int x = 0; x < weightDeltas.Length; x++) {
                        for (int y = 0; y < weightDeltas[x].Length; y++) {
                            dW[x][y] += deltaSet.weightsDelta[x][y];
                        }
                    }
                    dB += deltaSet.biasDelta;
                }

                // Average
                for (int x = 0; x < weightDeltas.Length; x++) {
                    for (int y = 0; y < weightDeltas[x].Length; y++) {
                        dW[x][y] /= Deltas.Count;
                    }
                }
                dB /= Deltas.Count;

                // Apply
                for (int x = 0; x < weightDeltas.Length; x++) {
                    for (int y = 0; y < weightDeltas[x].Length; y++) {
                        weights[x][y] -= dW[x][y] * NeuralNet.TrimMod; // TODO: Prob adjust trim mod based on reaction to cost after applying deltas
                    }
                }
                bias -= dB * NeuralNet.TrimMod;

                if (clearCache)
                    Deltas.Clear();
            }

            public double Activation(double a) => (Math.Pow(Math.E, a) - Math.Pow(Math.E, -a)) / (Math.Pow(Math.E, a) + Math.Pow(Math.E, -a));
            public double ActivationDer(double a) => 1 - Math.Pow(Activation(a), 2);
        }

        public struct TrainingDataDump {
            public double[] FedData;
            public double[] BeforeActivation;
            // public double[] AfterActivation;
        }

        public struct DeltaCache {
            public double biasDelta;
            public double[][] weightsDelta;
        }

    }
}