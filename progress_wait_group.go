package main

import (
	"sync"
	"sync/atomic"
)

type ProgressWaitGroup struct {
	wg        sync.WaitGroup
	total     int64
	completed int64
	mu        sync.Mutex // Protects access to completed
}

func (pwg *ProgressWaitGroup) Add(delta int) {
	pwg.wg.Add(delta)
	atomic.AddInt64(&pwg.total, int64(delta))
}

func (pwg *ProgressWaitGroup) Done() {
	pwg.wg.Done()
	pwg.mu.Lock()
	pwg.completed++
	pwg.mu.Unlock()
}

func (pwg *ProgressWaitGroup) Wait() {
	pwg.wg.Wait()
}

func (pwg *ProgressWaitGroup) Progress() float64 {
	pwg.mu.Lock()
	defer pwg.mu.Unlock()
	return float64(pwg.completed) / float64(pwg.total)
}
