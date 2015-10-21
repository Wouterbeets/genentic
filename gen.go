package gen

import (
	"fmt"
	"math/rand"
	"nn"
	"sort"
	"time"
)

type ai struct {
	*nn.Net
	score float64
	gene  []float64
}

type ByScore []*ai

func (ais ByScore) Len() int {
	return len(ais)
}

func (ais ByScore) Swap(i, j int) {
	ais[i], ais[j] = ais[j], ais[i]
}

func (ais ByScore) Less(i, j int) bool {
	return ais[i].score > ais[j].score
}

type Pool struct {
	ai        []*ai
	size      int
	roullete  [100]int
	mutatePer float64
	mStrength float64
}

func (p *Pool) Evolve(generations int, inp [][]float64, want []float64) {
	for i := 0; i < generations; i++ {
		p.Fight(inp, want)
		p.Breed()
	}
	sort.Sort(ByScore(p.ai))
	fmt.Printf("%.4f\n", p.ai[0].score)
	p.ai[0].In([]float64{0, 0})
	fmt.Printf("%.4f\n", p.ai[0].Out())
	p.ai[0].In([]float64{1, 0})
	fmt.Printf("%.4f\n", p.ai[0].Out())
	p.ai[0].In([]float64{0, 1})
	fmt.Printf("%.4f\n", p.ai[0].Out())
	p.ai[0].In([]float64{1, 1})
	fmt.Printf("%.4f\n", p.ai[0].Out())
}

func (p *Pool) String() (str string) {
	for _, ai := range p.ai {
		str += fmt.Sprintln(ai.score)
	}
	return
}

func (p *Pool) getSumScores() (sum float64) {
	for _, ai := range p.ai {
		sum += ai.score
	}
	return
}

func (p *Pool) makeRoullete() {
	sum := p.getSumScores()
	i := 0
	for k, ai := range p.ai {
		perc := (ai.score / sum) * float64(100)
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
	mGene := make([]float64, len(p.ai[m].gene))
	fGene := make([]float64, len(p.ai[f].gene))
	copy(mGene, p.ai[m].gene)
	copy(fGene, p.ai[f].gene)
	//fmt.Println(p.roullete)
	//	fmt.Println(m, f)
	//	fmt.Printf("%.2f\n", mGene)
	//	fmt.Printf("%.2f\n", fGene)
	baby = append(mGene[:len(mGene)/4], fGene[len(fGene)/4:]...)
	baby = append(baby[:(len(mGene)/4)*2], mGene[(len(fGene)/4)*2:]...)
	baby = append(baby[:(len(mGene)/4)*3], fGene[(len(fGene)/4)*3:]...)
	p.mutate(baby)
	//fmt.Printf("%.2f\n\n", baby)
	return
}

func (p *Pool) Breed() {
	sort.Sort(ByScore(p.ai))
	fmt.Printf("%.4f\n", p.ai[0].score)
	p.makeRoullete()
	for i := 3; i < p.size; i++ {
		m, f := p.roullete[rand.Intn(len(p.roullete))], p.roullete[rand.Intn(len(p.roullete))]
		if m == f {
			f = rand.Intn(p.size)
		}
		p.ai[i].SetWeights(p.makeBaby(m, f))
		p.ai[i].gene = p.ai[i].GetWeights()
	}
}

func abs(n float64) float64 {
	if n < 0 {
		return -n
	}
	return n
}

func fitnessFunc(resp, want float64) float64 {
	if want == 0 {
		if resp > 1 {
			return 0
		}
		return 1 - resp
	} else {
		return resp
	}
}

//TODO: generic fitness function
func (p *Pool) Fight(input [][]float64, want []float64) {
	for i, ai := range p.ai {
		score := float64(0)
		for j, v := range input {
			ai.In(v)
			dec := ai.Out()
			if dec[0] > 1 || dec[0] < 0 {
				panic("fuck")
			}
			score += fitnessFunc(dec[0], want[j])
		}
		p.ai[i].score = score
	}

}

func CreatePool(size int, mutatePer, mStrength float64, input, hidden, layers, output int) *Pool {
	pool := &Pool{
		ai:        make([]*ai, size),
		size:      size,
		mutatePer: mutatePer,
		mStrength: mStrength,
	}
	for i := range pool.ai {
		pool.ai[i] = &ai{
			nn.NewNet(input, hidden, layers, output),
			0,
			nil,
		}
		pool.ai[i].gene = pool.ai[i].GetWeights()
		pool.ai[i].Activate()
	}
	return pool
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
