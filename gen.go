/*
	Package gen allows you to manipulate a neural network using genetec generation of its weights.

	You create a pool of ai's using CreatePool. You must then assign a fitness function to the the pool
	This function will be used to assign scrores each generated neural network. The networks with the highest scores
	have the best chance of having their genes used in the next generartion.
*/
package gen

import (
	"fmt"
	names "github.com/Pallinder/go-randomdata"
	"github.com/Wouterbeets/nn"
	"log"
	"math/rand"
	"sort"
	"time"
)

const (
	ELITE = 3
)

type Challenge interface {
	Start(p1, p2 *Ai) (score1, score2 float64)
}

//Ai is a struct that holds a neural network, some varaibles to keep track of its perfomance.
type Ai struct {
	*nn.Net
	Score       float64
	TotalScore  float64
	GamesPlayed float64
	Name        string
}

//ByScore is a wrapper for a slice of ais and implements the sort interface
type ByTotalScore []*Ai

func (ais ByTotalScore) Len() int {
	return len(ais)
}

func (ais ByTotalScore) Swap(i, j int) {
	ais[i], ais[j] = ais[j], ais[i]
}

func (ais ByTotalScore) Less(i, j int) bool {
	return ais[i].TotalScore > ais[j].TotalScore
}

//ByScore is a wrapper for a slice of ais and implements the sort interface
type ByScore []*Ai

func (ais ByScore) Len() int {
	return len(ais)
}

func (ais ByScore) Swap(i, j int) {
	ais[i], ais[j] = ais[j], ais[i]
}

func (ais ByScore) Less(i, j int) bool {
	return ais[i].Score > ais[j].Score
}

//Pool is a struct that holds a slice of Ais.
type Pool struct {
	Ai        []*Ai
	size      int // number of ais
	roullete  [100]int
	mutatePer float64          //percentage of genes to be mutated
	mStrength float64          //how much the origninal value should be mutated by
	FightFunc func([]*Ai, int) //if function is set ,it will be used instead of standard Fight func. It must fill the ais Score field.
	Chal      Challenge
}

//Evolve runs the loop that aplies the genetic channges per generation
func (p *Pool) Evolve(generations int, inp [][]float64, want []float64) {
	for i := 0; i < generations; i++ {
		t := time.Now()
		log.Println("generation", i)
		if p.FightFunc != nil {
			p.FightFunc(p.Ai, i)
		} else if inp != nil && want != nil {
			p.Fight(inp, want)
		} else if p.Chal != nil {
			p.DoChal()
		}
		p.Breed()
		fmt.Println("estimated time to finish", time.Since(t)*time.Duration(generations-i))
	}
	sort.Sort(ByScore(p.Ai))
}

func (p *Pool) declareWinner(p1, p2 *Ai, s1, s2, s3, s4 float64) {
	p1.Score += (s1 + s4/2)
	p2.Score += (s2 + s3/2)
	p1.GamesPlayed += 1
	p2.GamesPlayed += 1
	p1.TotalScore += (s1 + s4) / 2
	p2.TotalScore += (s2 + s3) / 2
}

//tournament is a recursive function that creates a tree like tournament
//the winner of a match is retured by the function and then advances to the next round
//func (p *Pool) tournament(ais map[*Ai]int8, layerSize int) (winner *Ai) {
//	if layerSize >= len(ais) {
//		var ai1, ai2 *Ai
//		for ai1 = range ais {
//			delete(ais, ai1)
//			break
//		}
//		for ai2 = range ais {
//			delete(ais, ai2)
//			break
//		}
//		s1, s2 := p.Chal.Start(ai1, ai2)
//		s3, s4 := p.Chal.Start(ai2, ai1)
//		return p.declareWinner(ai1, ai2, s1, s2, s3, s4)
//	}
//	w1 := p.tournament(ais, layerSize*2)
//	w2 := p.tournament(ais, layerSize*2)
//	s1, s2 := p.Chal.Start(w1, w2)
//	s3, s4 := p.Chal.Start(w2, w1)
//	return p.declareWinner(w1, w2, s1, s2, s3, s4)
//}

func (p *Pool) DoChal() {
	for i := 0; i < len(p.Ai); {
		p1, j := p.Ai[i], i+1
		for j < len(p.Ai) {
			p2 := p.Ai[j]
			s1, s2 := p.Chal.Start(p1, p2)
			s3, s4 := p.Chal.Start(p2, p1)
			p.declareWinner(p1, p2, s1, s2, s3, s4)
			j++
		}
		i++
	}
	fmt.Println()
}

func (p *Pool) String() (str string) {
	for _, Ai := range p.Ai {
		str += fmt.Sprintln(Ai.Score)
	}
	return
}

func (p *Pool) getSumScores() (sum float64) {
	for _, Ai := range p.Ai {
		sum += Ai.Score
	}
	return
}

// makeRoullete creates an array filled with indexes of neural networks.
// the fittest neuralnetworks are more often represented in the array
// this array is later used to choose which neuralnetworks genes  from which we later randomly pick new parents
func (p *Pool) makeRoullete() {
	sum := p.getSumScores()
	i := 0
	for k, Ai := range p.Ai {
		perc := (Ai.Score / sum) * float64(100)
		for j := 0; j < int(perc); i, j = i+1, j+1 {
			p.roullete[i] = k
		}
	}
}

func (p *Pool) mutate(genes []float64) {
	gToMutate := float64(len(genes)) * p.mutatePer
	if gToMutate < 1 {
		gToMutate = 1
	}
	for i := 0; i < int(gToMutate); i++ {
		r := rand.Int()
		if r%2 == 0 {
			genes[rand.Intn(len(genes))] += rand.NormFloat64() * p.mStrength
		} else {
			genes[rand.Intn(len(genes))] -= rand.NormFloat64() * p.mStrength
		}
	}
}

func (p *Pool) makeBaby(m, f int) (baby []float64) {
	mGene := p.Ai[m].GetWeights()
	fGene := p.Ai[f].GetWeights()
	nSplits := rand.Intn((len(mGene) / 2)) + 2
	splits := make([]int, nSplits, nSplits+1)
	for i := 1; i < nSplits; i++ {
		splits[i] = rand.Intn(len(mGene))
	}
	splits = append(splits, len(mGene))
	sort.Sort(sort.IntSlice(splits))
	baby = make([]float64, 0, len(mGene))
	parent := mGene
	dad := true
	for i := 0; i < len(splits)-1; i++ {
		if splits[i] != splits[i+1] {
			baby = append(baby, parent[splits[i]:splits[i+1]]...)
			if dad {
				parent = fGene
				dad = false
			} else {
				dad = true
				parent = mGene
			}
		}
	}
	p.mutate(baby)
	return
}

func (p *Pool) PrintTopStats(prefix string, top int) {
	for i, ai := range p.Ai {
		_ = i
		if i < top {
			fmt.Printf("%s -- %20s : %6.2f : %6.2f : %6.2f\n", prefix, ai.Name, ai.Score, ai.TotalScore, ai.TotalScore/ai.GamesPlayed)
		}
	}
	log.Println("\n")
}

func (p *Pool) resetScores() {
	for _, ai := range p.Ai {
		ai.Score = 0
	}
}
func (p *Pool) printStats() {
	sort.Sort(ByTotalScore(p.Ai))
	p.PrintTopStats("total", 5)
	sort.Sort(ByScore(p.Ai))
	p.PrintTopStats("gener", 5)
	log.Println("\n")
}

//Breed make new ai from the best ai this genereation
func (p *Pool) Breed() {
	sort.Sort(ByScore(p.Ai))
	p.makeRoullete()
	for i := ELITE; i < p.size; i++ {
		f, m := 0, 0
		if rand.Int()%2 == 0 {
			m = i
			f = p.roullete[rand.Intn(len(p.roullete))]
		} else {
			f = i
			m = p.roullete[rand.Intn(len(p.roullete))]
		}
		for m == f {
			f = rand.Intn(p.size)
		}
		p.Ai[i].SetWeights(p.makeBaby(m, f))
	}
	p.printStats()
	p.resetScores()
}

func abs(n float64) float64 {
	if n < 0 {
		return -n
	}
	return n
}

func fitnessFunc(resp, want float64) float64 {
	if want == 0 {
		return 1 - resp
	}
	return resp
}

//Fight is the standard fitness func for whne you want to create dumb things
func (p *Pool) Fight(input [][]float64, want []float64) {
	for i, Ai := range p.Ai {
		Score := float64(0)
		for j, v := range input {
			go Ai.In(v)
			dec := Ai.Out()
			Score += fitnessFunc(dec[0], want[j])
		}
		p.Ai[i].Score = Score
	}
}

//CreatePool is the constructor of the pool of ais that are to be evolved
func CreatePool(size int, mutatePer, mStrength float64, input, hidden, layers, output int) *Pool {
	pool := &Pool{
		Ai:        make([]*Ai, size),
		size:      size,
		mutatePer: mutatePer,
		mStrength: mStrength,
		FightFunc: nil,
	}
	for i := range pool.Ai {
		pool.Ai[i] = &Ai{
			nn.NewNet(input, hidden, layers, output),
			0,
			0,
			0,
			names.SillyName(),
		}
	}
	return pool
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
