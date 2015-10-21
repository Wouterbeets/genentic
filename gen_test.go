package gen

import (
	"fmt"
	"testing"
)

func TestFitness(t *testing.T) {
	var tests = []struct {
		in   float64
		want float64
	}{
		{
			in:   0.01,
			want: 0,
		}, {
			in:   0.5,
			want: 0,
		}, {
			in:   0.99,
			want: 0,
		}, {
			in:   0.01,
			want: 1,
		}, {
			in:   0.5,
			want: 1,
		}, {
			in:   0.99,
			want: 1,
		},
	}
	for _, test := range tests {
		fmt.Println(fitnessFunc(test.in, test.want))
	}
}

func TestCreatePool(t *testing.T) {
	var tests = []struct {
		poolSize    int
		generations int
		mutatePer   float64
		mStrenght   float64
		inpNeur     int
		hiddenNeur  int
		totalLayers int
		outNeur     int
		inp         [][]float64
		want        []float64
	}{
		{
			poolSize:    10,
			generations: 50,
			mutatePer:   0.2,
			mStrenght:   10,
			inpNeur:     2,
			hiddenNeur:  5,
			totalLayers: 4,
			outNeur:     1,
			inp: [][]float64{
				{0, 0},
				{1, 0},
				{0, 1},
				{1, 1},
			},
			want: []float64{
				0, 1, 1, 0,
			},
		},

		{
			poolSize:    10,
			generations: 200,
			mutatePer:   0.2,
			mStrenght:   10,
			inpNeur:     2,
			hiddenNeur:  2,
			totalLayers: 3,
			outNeur:     1,
			inp: [][]float64{
				{0, 0},
				{1, 0},
				{0, 1},
				{1, 1},
			},
			want: []float64{
				0, 1, 1, 0,
			},
		}, {
			poolSize:    10,
			generations: 300,
			mutatePer:   0.2,
			mStrenght:   10,
			inpNeur:     2,
			hiddenNeur:  2,
			totalLayers: 3,
			outNeur:     1,
			inp: [][]float64{
				{0, 0},
				{1, 0},
				{0, 1},
				{1, 1},
			},
			want: []float64{
				0, 1, 1, 0,
			},
		}, {
			poolSize:    100,
			generations: 100,
			mutatePer:   0.2,
			mStrenght:   10,
			inpNeur:     2,
			hiddenNeur:  2,
			totalLayers: 3,
			outNeur:     1,
			inp: [][]float64{
				{0, 0},
				{1, 0},
				{0, 1},
				{1, 1},
			},
			want: []float64{
				0, 1, 1, 0,
			},
		}, {
			poolSize:    100,
			generations: 200,
			mutatePer:   0.2,
			mStrenght:   10,
			inpNeur:     2,
			hiddenNeur:  2,
			totalLayers: 3,
			outNeur:     1,
			inp: [][]float64{
				{0, 0},
				{1, 0},
				{0, 1},
				{1, 1},
			},
			want: []float64{
				0, 1, 1, 0,
			},
		},

		{
			poolSize:    100,
			generations: 1000,
			mutatePer:   0.2,
			mStrenght:   10,
			inpNeur:     2,
			hiddenNeur:  4,
			totalLayers: 3,
			outNeur:     1,
			inp: [][]float64{
				{0, 0},
				{1, 0},
				{0, 1},
				{1, 1},
			},
			want: []float64{
				0, 1, 1, 0,
			},
		},

		{
			poolSize:    200,
			generations: 100,
			mutatePer:   0.2,
			mStrenght:   10,
			inpNeur:     2,
			hiddenNeur:  2,
			totalLayers: 3,
			outNeur:     1,
			inp: [][]float64{
				{0, 0},
				{1, 0},
				{0, 1},
				{1, 1},
			},
			want: []float64{
				0, 1, 1, 0,
			},
		},

		{
			poolSize:    200,
			generations: 200,
			mutatePer:   0.2,
			mStrenght:   10,
			inpNeur:     2,
			hiddenNeur:  2,
			totalLayers: 3,
			outNeur:     1,
			inp: [][]float64{
				{0, 0},
				{1, 0},
				{0, 1},
				{1, 1},
			},
			want: []float64{
				0, 1, 1, 0,
			},
		}, {
			poolSize:    200,
			generations: 300,
			mutatePer:   0.2,
			mStrenght:   10,
			inpNeur:     2,
			hiddenNeur:  2,
			totalLayers: 3,
			outNeur:     1,
			inp: [][]float64{
				{0, 0},
				{1, 0},
				{0, 1},
				{1, 1},
			},
			want: []float64{
				0, 1, 1, 0,
			},
		},
	}
	for _, test := range tests {
		p := CreatePool(test.poolSize, test.mutatePer, test.mStrenght, test.inpNeur, test.hiddenNeur, test.totalLayers, test.outNeur)
		fmt.Println("poolsize\t", test.poolSize, "\tgenerations\t", test.generations, "\tmutation\t", test.mutatePer)
		p.Evolve(test.generations, test.inp, test.want)
	}
}
