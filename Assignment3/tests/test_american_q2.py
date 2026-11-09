"""Independent CRR/discrete-dividend tests, including an exhaustive small tree."""
from functools import lru_cache
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assignment3.american import exercise_boundary, tree_price
from assignment3.pricing import price


def exhaustive(s, k, t, r, sigma, n, call, american, dividends):
    """Independent path recursion with no node merging or event subtrees."""
    dt=t/n;u=np.exp(sigma*np.sqrt(dt));d=1/u;p=(np.exp(r*dt)-d)/(u-d)
    events={int(round(when/dt)):cash for when,cash in dividends}
    sign=1 if call else -1
    def value(step, spot):
        pre=max(sign*(spot-k),0.)
        if step==n:return pre
        after=max(spot-events.get(step,0.),0.)
        continuation=np.exp(-r*dt)*(p*value(step+1,after*u)+(1-p)*value(step+1,after*d))
        return max(pre,max(sign*(after-k),0.),continuation) if american else continuation
    return value(0,s)


@pytest.mark.parametrize('call',[False,True])
@pytest.mark.parametrize('american',[False,True])
@pytest.mark.parametrize('dividends',[(),((.1,.25),),((.1,.25),(.4,1.5))])
def test_event_subtrees_equal_exhaustive_small_nonrecombining_tree(call,american,dividends):
    args=(48.5,50.,.5,.045,.3,5,call,american,dividends)
    actual=tree_price(*args[:6],is_call=call,american=american,dividends=dividends)
    assert actual==pytest.approx(exhaustive(*args),abs=1e-12)


@pytest.mark.parametrize('call',[False,True])
def test_zero_dividend_european_converges_to_gbsm(call):
    analytic=price(48.5,50,.5,.045,.045,.3,call)
    lattice=tree_price(48.5,50,.5,.045,.3,1000,call,False)
    assert abs(lattice-analytic)<.002


@pytest.mark.parametrize('n',[25,100,105,200])
def test_no_dividend_american_call_has_no_exercise_premium(n):
    european=tree_price(48.5,50,.5,.045,.3,n,True,False)
    american=tree_price(48.5,50,.5,.045,.3,n,True,True)
    assert american==pytest.approx(european,abs=1e-11)


@pytest.mark.parametrize('n',[25,100,105,200])
def test_dividend_european_parity_and_american_bounds(n):
    divs=((.1,.25),(.4,1.5));s,k,t,r,v=48.5,50.,.5,.045,.3
    ec=tree_price(s,k,t,r,v,n,True,False,divs)
    ep=tree_price(s,k,t,r,v,n,False,False,divs)
    ac=tree_price(s,k,t,r,v,n,True,True,divs)
    ap=tree_price(s,k,t,r,v,n,False,True,divs)
    parity=s-sum(cash*np.exp(-r*when) for when,cash in divs)-k*np.exp(-r*t)
    assert ec-ep==pytest.approx(parity,abs=1e-10)
    assert ac>=ec and ap>=ep
    assert ac<=s and ap<=k


def test_dividend_and_spot_monotonicity_and_vectorization():
    divs=((.1,.25),(.4,1.5));spots=np.array([45.,48.5,52.])
    calls=tree_price(spots,50,.5,.045,.3,100,True,True,divs)
    puts=tree_price(spots,50,.5,.045,.3,100,False,True,divs)
    assert np.all(np.diff(calls)>0) and np.all(np.diff(puts)<0)
    for i,s in enumerate(spots):
        assert calls[i]==pytest.approx(tree_price(s,50,.5,.045,.3,100,True,True,divs))
    higher=((.1,.25),(.4,1.6))
    assert tree_price(48.5,50,.5,.045,.3,100,True,True,higher)<=calls[1]
    assert tree_price(48.5,50,.5,.045,.3,100,False,True,higher)>=puts[1]


def test_exercise_boundary_changes_decision_on_each_side():
    boundary=exercise_boundary(50,.1,.045,.3,40,1.5)
    assert boundary>50
    for offset in [-.01,.01]:
        s=boundary+offset
        continue_value=tree_price(s-1.5,50,.1,.045,.3,40,True,True)
        difference=max(s-50,0)-continue_value
        assert np.sign(difference)==np.sign(offset)
    assert exercise_boundary(50,.4,.045,.3,80,.25) is None
    closed_form_boundary=optimize.brentq(lambda s:s-50-price(s-1.5,50,.1,.045,.045,.3,True),50,80)
    assert abs(exercise_boundary(50,.1,.045,.3,200,1.5)-closed_form_boundary)<.003


def test_small_fixed_grid_gamma_vanishes_away_from_kinks():
    s,k,t,r,v,n=48.5,50.,.5,.045,.3,100;divs=((.1,.25),(.4,1.5));h=1e-5
    base=tree_price(s,k,t,r,v,n,True,True,divs)
    up=tree_price(s+h,k,t,r,v,n,True,True,divs)
    down=tree_price(s-h,k,t,r,v,n,True,True,divs)
    assert abs((up-2*base+down)/h**2)<.001


@pytest.mark.parametrize('kwargs',[
    dict(n=102,dividends=((.1,.25),)),dict(n=0),dict(s=-1),dict(sigma=0),
    dict(dividends=((.1,-.25),)),dict(dividends=((.5,.25),)),dict(is_call='call'),
])
def test_invalid_tree_inputs_rejected(kwargs):
    arguments=dict(s=48.5,k=50.,t=.5,r=.045,sigma=.3,n=100,is_call=True,american=True)
    arguments.update(kwargs)
    with pytest.raises(ValueError):tree_price(**arguments)
