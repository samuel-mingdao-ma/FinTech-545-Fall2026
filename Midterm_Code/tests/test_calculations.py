"""Numerical tests on synthetic inputs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from calculations import q1,q2,q3b,q3c,inverse_margin
from examkit import Fit,fit_distribution


def test_q1_pairwise_variances_and_psd(tmp_path):
    data=tmp_path/'data';out=tmp_path/'out';data.mkdir();out.mkdir()
    # Deliberately incompatible observed pairs produce an indefinite correlation.
    x=pd.DataFrame({'x1':[1,2,1,2,np.nan,np.nan],
                    'x2':[1,2,np.nan,np.nan,1,2],
                    'x3':[np.nan,np.nan,1,2,2,1]})
    x.to_csv(data/'problem1.csv',index=False)
    q1(data,out)
    original=pd.read_csv(out/'q1a_pairwise_correlation.csv',index_col=0).to_numpy()
    np.testing.assert_allclose(original,[[1,1,1],[1,1,-1],[1,-1,1]])
    repaired=pd.read_csv(out/'q1d_higham_correlation.csv',index_col=0).to_numpy()
    assert np.linalg.eigvalsh(repaired).min()>-1e-9
    np.testing.assert_allclose(np.diag(repaired),1)
    cov=pd.read_csv(out/'q1e_covariance.csv',index_col=0).to_numpy()
    np.testing.assert_allclose(np.diag(cov),x.var(ddof=1))


def test_q2_risk_against_sample_normal(tmp_path):
    data=tmp_path/'data';out=tmp_path/'out';data.mkdir();out.mkdir()
    rng=np.random.default_rng(99)
    spy=rng.normal(.0001,.01,500)
    xyz=.0002+1.3*spy+rng.normal(0,.02,500)
    p=pd.DataFrame({'Date':np.arange(501),'SPY':np.r_[100,100*np.cumprod(1+spy)],'XYZ':np.r_[80,80*np.cumprod(1+xyz)]})
    p.to_csv(data/'problem2.csv',index=False)
    q2(data,out)
    result=pd.read_csv(out/'q2_risk_percent.csv')
    z=stats.norm.ppf([.05,.01])
    expected=-xyz.mean()-xyz.std(ddof=1)*z
    np.testing.assert_allclose(result.VaR_percent,100*expected,rtol=1e-10)
    expected_es=-xyz.mean()+xyz.std(ddof=1)*stats.norm.pdf(z)/[.05,.01]
    np.testing.assert_allclose(result.ES_percent,100*expected_es,rtol=1e-10)


def test_nig_inverse_interpolation_probability_accuracy():
    d=stats.norminvgauss(2,-.4,loc=.02,scale=.05)
    f=Fit('nig',d,dict(mu=.02,alpha=40,beta=-8,delta=.05),0,0,0)
    u=np.array([1e-6,.001,.01,.05,.5,.95,.99,.999,1-1e-6])
    values,error=inverse_margin(f,u)
    np.testing.assert_allclose(d.cdf(values),u,atol=1e-5)
    assert error<1e-5


@pytest.mark.parametrize('copula',['gaussian','t'])
def test_q3_explicit_copula_synthetic_holdings_and_books(tmp_path,copula):
    data=tmp_path/'data';out=tmp_path/'out';data.mkdir();out.mkdir()
    rng=np.random.default_rng(33)
    names=[f'S{i}' for i in range(1,11)]
    factor=rng.normal(0,.01,(250,1))
    r=pd.DataFrame(factor+rng.normal(0,.01,(250,10)),columns=names)
    prices=pd.DataFrame(np.vstack([np.ones(10)*100,100*np.cumprod(1+r,axis=0)]),columns=names)
    pd.DataFrame({'Stock':names,'Book':['A']*5+['B']*5,'Holding':[10]*10}).to_csv(data/'problem3_holdings.csv',index=False)
    # Fixed Normal margins isolate the copula and aggregation calculations.
    choices={s:'normal' for s in names}
    fits={s:{'normal':fit_distribution(r[s],'normal',normal_sample_std=False)} for s in names}
    c=q3b(out,r,fits,choices)
    q3c(data,out,prices,r,fits,choices,5000,545,c,copula)
    scores=pd.read_csv(out/'q3b_copula_scores.csv')
    assert set(scores.copula)=={'gaussian','t'}
    risk=pd.read_csv(out/'q3c_risk.csv')
    assert len(risk)==13
    assert set(risk.copula)=={copula}
    assert set(risk.position)==set(names+['Book A','Book B','Total'])
    assert np.isfinite(risk[['VaR95_dollars','ES95_dollars']]).all().all()


@pytest.mark.parametrize('copula',['gaussian','t'])
def test_q3_mixed_synthetic_margins(tmp_path,copula):
    data=tmp_path/'data';out=tmp_path/'out';data.mkdir();out.mkdir()
    rng=np.random.default_rng(734)
    names=['SYN_NORMAL','SYN_T','SYN_NIG']
    distributions=[stats.norm(.001,.015),stats.t(6,loc=.002,scale=.012),
                   stats.norminvgauss(2.2,-.5,loc=.003,scale=.035)]
    families=['normal','t','nig']
    params=[dict(mu=.001,sigma=.015),dict(mu=.002,scale=.012,nu=6),
            dict(mu=.003,alpha=2.2/.035,beta=-.5/.035,delta=.035)]
    r=pd.DataFrame({s:d.rvs(size=300,random_state=rng) for s,d in zip(names,distributions)})
    prices=pd.DataFrame(np.vstack([np.ones(3)*100,100*np.cumprod(1+r,axis=0)]),columns=names)
    choices=dict(zip(names,families))
    fits={s:{family:Fit(family,d,p,0,0,0)}
          for s,family,d,p in zip(names,families,distributions,params)}
    pd.DataFrame(dict(Stock=names,Book=['A','A','B'],Holding=[10,20,15])).to_csv(
        data/'problem3_holdings.csv',index=False)
    c=q3b(out,r,fits,choices)
    q3c(data,out,prices,r,fits,choices,5000,545,c,copula)
    risk=pd.read_csv(out/'q3c_risk.csv')
    assert choices==dict(zip(names,families))
    assert len(risk)==6 and set(risk.copula)=={copula}
    assert np.isfinite(risk[['VaR95_dollars','ES95_dollars']]).all().all()
    checks=pd.read_csv(out/'q3c_marginal_validation.csv')
    assert checks.z.abs().max()<7
    interpolation=pd.read_csv(out/'q3c_nig_inverse_checks.csv')
    assert interpolation.cdf_interpolation_max_error.max()<1e-5
