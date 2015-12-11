package gen

import (
	"encoding/gob"
	"fmt"
	"math/rand"
	"github.com/WouterBeets/nn"
	"os"
	"sort"
	"time"
)

type Ai struct {
	*nn.Net
	Score       float64
	GamesPlayed float64
	gene        []float64
}

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

type Pool struct {
	Ai        []*Ai
	size      int
	roullete  [100]int
	mutatePer float64
	mStrength float64
	FightFunc func([]*Ai, int) //if function is set ,it will be used instead of standard Fight func. It must fill the ais Score field.
}

func (p *Pool) Evolve(generations int, inp [][]float64, want []float64) {
	file, err := os.OpenFile("foo.gob", os.O_RDWR|os.O_APPEND, 0660)
	if err != nil {
		file, err = os.Create("foo.gob")
	}
	saver := gob.NewEncoder(file)
	for i := 0; i < generations; i++ {
		fmt.Println("generation", i)
		if i == 299000 {
			str := ""
			fmt.Scanln(&str)
		}
		if p.FightFunc != nil {
			p.FightFunc(p.Ai, i)
		} else if inp != nil && want != nil {
			p.Fight(inp, want)
		}
		p.Breed()
		saver.Encode(p.Ai[0].Net.GetWeights())
	}
	sort.Sort(ByScore(p.Ai))
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

//makeRoullete makes sure that the fittest individuals are more often bred by filling the breeding array more often with their index, from which we later randomly pick new parents
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
	mGene := make([]float64, len(p.Ai[m].gene))
	fGene := make([]float64, len(p.Ai[f].gene))
	copy(mGene, p.Ai[m].gene)
	copy(fGene, p.Ai[f].gene)
	baby = append(mGene[:len(mGene)/4], fGene[len(fGene)/4:]...)
	baby = append(baby[:(len(mGene)/4)*2], mGene[(len(fGene)/4)*2:]...)
	baby = append(baby[:(len(mGene)/4)*3], fGene[(len(fGene)/4)*3:]...)
	p.mutate(baby)
	return
}

func (p *Pool) Breed() {
	sort.Sort(ByScore(p.Ai))
	//fmt.Printf("%.4f\n", p.Ai[0].Score)
	p.makeRoullete()
	for i := 1; i < p.size; i++ {
		m, f := p.roullete[rand.Intn(len(p.roullete))], p.roullete[rand.Intn(len(p.roullete))]
		if m == f {
			f = rand.Intn(p.size)
		}
		//fmt.Printf("%.4f\n", p.Ai[i].Score)
		p.Ai[i].SetWeights(p.makeBaby(m, f))
		p.Ai[i].gene = p.Ai[i].GetWeights()
	}
	for i, ai := range p.Ai {
		if i < 5 {
			fmt.Println(ai.Score)
		}
		ai.Score = 0
		ai.GamesPlayed = 0
	}
	fmt.Println("\n")
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
	} else {
		return resp
	}
}

func (p *Pool) Fight(input [][]float64, want []float64) {
	for i, Ai := range p.Ai {
		Score := float64(0)
		for j, v := range input {
			Ai.In(v)
			dec := Ai.Out()
			Score += fitnessFunc(dec[0], want[j])
		}
		p.Ai[i].Score = Score
	}
}

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
			nil,
		}
		pool.Ai[i].gene = pool.Ai[i].GetWeights()
		pool.Ai[i].Activate()
	}
	return pool
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
