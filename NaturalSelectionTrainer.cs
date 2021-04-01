using System;
using System.Collections.Generic;

namespace BigBrian {
    public class NaturalSelectionTrainer {

        private int[] structure;
        private int sampleSize;
        private List<NeuralNet> networks = new List<NeuralNet>();

        public NaturalSelectionTrainer(int sampleSize, int[] structure) {
            this.sampleSize = sampleSize;
            this.structure = structure;
            while (networks.Count < sampleSize) {
                networks.Add(new NeuralNet(structure, 7654383 + DateTime.Now.Millisecond * networks.Count));
            }
        }

        public void TestGeneration((double[] input, double[] output)[] data) {
            foreach (var n in networks) {
                foreach (var d in data) {
                    n.Passthrough(d.input, d.output);
                }
            }

            networks = MergeSortNetworks(networks);
        }

        public void Repopulate() {
            while (networks.Count < sampleSize) {
                networks.Add(new NeuralNet(structure, 7654383 + DateTime.Now.Millisecond * networks.Count));
            }
        }

        public List<NeuralNet> MergeSortNetworks(List<NeuralNet> a) => MergeSortNetworks(a.GetRange(0, a.Count / 2), a.GetRange(a.Count / 2, a.Count - (a.Count / 2)));

        public List<NeuralNet> MergeSortNetworks(List<NeuralNet> a, List<NeuralNet> b) {
            if (a.Count > 1)
                a = MergeSortNetworks(a.GetRange(0, a.Count / 2), a.GetRange(a.Count / 2, a.Count - (a.Count / 2)));
            if (b.Count > 1)
                b = MergeSortNetworks(b.GetRange(0, b.Count / 2), b.GetRange(b.Count / 2, b.Count - (b.Count / 2)));

            List<NeuralNet> sortedList = new List<NeuralNet>();
            int counterA = 0;
            int counterB = 0;

            while (sortedList.Count < a.Count + b.Count) {

                if (counterA >= a.Count) {
                    sortedList.Add(b[counterB]);
                    counterB++;
                } else if (counterB >= b.Count) {
                    sortedList.Add(a[counterA]);
                    counterA++;
                } else {
                    if (a[counterA].Cost > b[counterB].Cost) {
                        sortedList.Add(b[counterB]);
                        counterB++;
                    } else {
                        sortedList.Add(a[counterA]);
                        counterA++;
                    }
                }
            }

            return sortedList;
        }
    }
}