
# Large neighborhood search

### Algorithm 1: Large neighborhood search


```
1: input: a feasible solution x
2: x_best = x;
3: repeat
4:     x_temp = repair(destroy(x));
5:     if accept(x_temp, x) then
6:         x = x_temp;
7:     end if
8:     if cost(x_temp) < cost(x_best) then
9:         x_best = x_temp;
10:    end if
11: until stopping criterion is met
12: return x_best
```