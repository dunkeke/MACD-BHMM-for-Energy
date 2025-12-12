import numpy as np
import pandas as pd
import streamlit as st
import pandas_ta as ta
import yfinance as yf
from scipy.signal import find_peaks
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedMACD_HMM_Strategy:
    """
    增强型MACD-HMM能源期货交易策略
    结合MACD多形态识别与HMM市场状态检测
    """
    
    def __init__(self, data, fast_period=12, slow_period=26, signal_period=9,
                 n_hmm_states=4, n_hmm_features=8):
        """
        初始化策略
        
        Parameters:
        -----------
        data : DataFrame
            能源期货数据（原油、天然气等）
        fast_period, slow_period, signal_period : int
            MACD参数
        n_hmm_states : int
            HMM市场状态数量
        n_hmm_features : int
            HMM特征维度
        """
        self.data = data.copy()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.n_hmm_states = n_hmm_states
        self.n_hmm_features = n_hmm_features
        
        # 特征工程
        self.prepare_features()
        
        # 计算MACD技术指标
        self.calculate_macd_features()
        
        # 训练HMM模型
        self.train_hmm_model()
        
        # 生成交易信号
        self.generate_trading_signals()
        
        # 策略绩效分析
        self.analyze_performance()
    
    def prepare_features(self):
        """准备能源期货特征"""
        df = self.data
        
        # 基础价格特征
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        df['close_open_spread'] = (df['close'] - df['open']) / df['open']
        
        # 波动率特征
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # 成交量特征（如果有且为数值型）
        if 'volume' in df.columns and pd.api.types.is_numeric_dtype(df['volume']):
            volume_series = pd.to_numeric(df['volume'], errors='coerce')
            df['volume_ma'] = volume_series.rolling(20).mean()
            df['volume_ratio'] = volume_series / df['volume_ma']
            df['price_volume_corr'] = df['close'].rolling(10).corr(volume_series)
        
        # RSI指标
        df['rsi'] = ta.rsi(df['close'], length=14)

        # 布林带（兼容不同版本/列名，且在计算失败时跳过）
        # pandas-ta 部分环境可能返回 None 或列名变更，异常时直接跳过以避免 Streamlit 崩溃
        try:
            bbands = ta.bbands(df['close'], length=20, std=2)
        except Exception:
            bbands = None

        if isinstance(bbands, pd.DataFrame):
            upper_col = next((c for c in bbands.columns if c.startswith('BBU_')), None)
            middle_col = next((c for c in bbands.columns if c.startswith('BBM_')), None)
            lower_col = next((c for c in bbands.columns if c.startswith('BBL_')), None)

            if upper_col and middle_col and lower_col:
                df['bb_upper'] = bbands[upper_col]
                df['bb_middle'] = bbands[middle_col]
                df['bb_lower'] = bbands[lower_col]
                spread = df['bb_upper'] - df['bb_lower']
                df['bb_position'] = np.where(spread != 0, (df['close'] - df['bb_lower']) / spread, np.nan)
        
        # ATR（平均真实波幅）- 兼容 pandas-ta 返回 None 或长度不匹配的情况
        atr_series = None
        try:
            atr_candidate = ta.atr(df['high'], df['low'], df['close'], length=14)
            if atr_candidate is not None and len(atr_candidate) == len(df):
                atr_series = pd.to_numeric(atr_candidate, errors='coerce')
                # 确保索引对齐，防止赋值长度错误
                atr_series = pd.Series(atr_series.values, index=df.index)
        except Exception:
            atr_series = None

        if atr_series is not None:
            df['atr'] = atr_series
            df['atr_ratio'] = np.where(df['close'] != 0, df['atr'] / df['close'], np.nan)

        # 移动平均线
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_60'] = df['close'].rolling(60).mean()
        df['ma_cross'] = np.where(df['ma_5'] > df['ma_20'], 1, -1)
        
        # 能源特有特征
        df['seasonal_factor'] = self.calculate_seasonal_factor(df.index)
        
        self.data = df.dropna()
    
    def calculate_seasonal_factor(self, dates):
        """计算能源季节性因子（原油、天然气有季节性特征）"""
        seasonal = np.zeros(len(dates))
        for i, date in enumerate(dates):
            month = date.month
            # 冬季（天然气需求高）
            if month in [11, 12, 1, 2]:
                seasonal[i] = 1.0
            # 夏季（原油需求高）
            elif month in [5, 6, 7, 8]:
                seasonal[i] = 0.5
            else:
                seasonal[i] = 0.0
        return seasonal
    
    def calculate_macd_features(self):
        """计算MACD多形态特征"""
        close_prices = self.data['close'].values
        
        try:
            macd = ta.macd(
                close_prices,
                fast=self.fast_period,
                slow=self.slow_period,
                signal=self.signal_period
            )
        except Exception:
            macd = None

        # 为非标准/失败输出提供容错，避免缺列引发 KeyError
        macd_len = len(self.data)
        dif_col = f"MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}"
        dea_col = f"MACDs_{self.fast_period}_{self.slow_period}_{self.signal_period}"
        hist_col = f"MACDh_{self.fast_period}_{self.slow_period}_{self.signal_period}"

        def _safe_get(col_name, fallback_prefix=None):
            if isinstance(macd, pd.DataFrame) and len(macd) == macd_len:
                if col_name in macd.columns:
                    return macd[col_name]
                if fallback_prefix:
                    alt = next((c for c in macd.columns if c.startswith(fallback_prefix)), None)
                    if alt:
                        return macd[alt]
            return pd.Series(np.nan, index=self.data.index)

        self.data['DIF'] = _safe_get(dif_col, fallback_prefix='MACD_')
        self.data['DEA'] = _safe_get(dea_col, fallback_prefix='MACDs_')
        self.data['MACD_hist'] = _safe_get(hist_col, fallback_prefix='MACDh_')
        macd = ta.macd(
            close_prices,
            fast=self.fast_period,
            slow=self.slow_period,
            signal=self.signal_period
        )

        self.data['DIF'] = macd[f"MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}"]
        self.data['DEA'] = macd[f"MACDs_{self.fast_period}_{self.slow_period}_{self.signal_period}"]
        self.data['MACD_hist'] = macd[f"MACDh_{self.fast_period}_{self.slow_period}_{self.signal_period}"]
        
        # 1. 基本交叉信号
        self.data['golden_cross'] = (self.data['DIF'] > self.data['DEA']) & \
                                   (self.data['DIF'].shift(1) <= self.data['DEA'].shift(1))
        self.data['death_cross'] = (self.data['DIF'] < self.data['DEA']) & \
                                  (self.data['DIF'].shift(1) >= self.data['DEA'].shift(1))
        
        # 2. 水下信号
        self.data['under_water'] = self.data['DIF'] < 0
        self.data['underwater_golden'] = self.data['golden_cross'] & self.data['under_water']
        self.data['underwater_death'] = self.data['death_cross'] & self.data['under_water']
        
        # 3. 双波峰波谷检测
        self.data['double_peak'], self.data['double_valley'] = self.detect_double_patterns(
            self.data['DIF'].values, window=10
        )
        
        # 4. MACD动量特征
        self.data['macd_momentum'] = self.data['DIF'].diff(3)
        self.data['macd_acceleration'] = self.data['macd_momentum'].diff(3)
        self.data['hist_trend'] = self.data['MACD_hist'].rolling(5).mean()
        
        # 5. 背离检测
        self.data['price_macd_divergence'] = self.detect_divergence(
            self.data['close'], self.data['DIF']
        )
        
        # 6. MACD强度指标
        self.data['macd_strength'] = self.calculate_macd_strength()

    def build_macd_event_log(self):
        """输出MACD事件表，便于调试和回测信号出现的位置"""
        event_columns = [
            'golden_cross',
            'death_cross',
            'underwater_golden',
            'underwater_death',
            'double_peak',
            'double_valley'
        ]

        existing_cols = [col for col in event_columns if col in self.data.columns]
        return self.data[existing_cols].astype(bool)
    
    def detect_double_patterns(self, series, window=10):
        """检测双波峰和双波谷形态"""
        n = len(series)
        double_peak = np.zeros(n, dtype=bool)
        double_valley = np.zeros(n, dtype=bool)
        
        # 寻找局部极值点
        peaks, _ = find_peaks(series, distance=window//2, prominence=np.std(series)/10)
        valleys, _ = find_peaks(-series, distance=window//2, prominence=np.std(series)/10)
        
        # 检测双波峰
        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i-1] <= window:
                # 检查第二个波峰是否低于第一个
                if series[peaks[i]] < series[peaks[i-1]]:
                    double_peak[peaks[i]] = True
        
        # 检测双波谷
        for i in range(1, len(valleys)):
            if valleys[i] - valleys[i-1] <= window:
                # 检查第二个波谷是否高于第一个
                if series[valleys[i]] > series[valleys[i-1]]:
                    double_valley[valleys[i]] = True
        
        return double_peak, double_valley
    
    def detect_divergence(self, price, indicator, window=20):
        """检测价格与指标的背离"""
        divergence = np.zeros(len(price))
        
        # 寻找价格和指标的局部极值
        price_peaks, _ = find_peaks(price, distance=window//2)
        price_valleys, _ = find_peaks(-price, distance=window//2)
        ind_peaks, _ = find_peaks(indicator, distance=window//2)
        ind_valleys, _ = find_peaks(-indicator, distance=window//2)
        
        # 顶背离：价格新高，指标新低
        if len(price_peaks) > 1 and len(ind_peaks) > 1:
            if price[price_peaks[-1]] > price[price_peaks[-2]] and \
               indicator[ind_peaks[-1]] < indicator[ind_peaks[-2]]:
                divergence[price_peaks[-1]] = -1  # 看跌信号
        
        # 底背离：价格新低，指标新高
        if len(price_valleys) > 1 and len(ind_valleys) > 1:
            if price[price_valleys[-1]] < price[price_valleys[-2]] and \
               indicator[ind_valleys[-1]] > indicator[ind_valleys[-2]]:
                divergence[price_valleys[-1]] = 1  # 看涨信号
        
        return divergence
    
    def calculate_macd_strength(self):
        """计算MACD综合强度"""
        df = self.data
        
        # 标准化各维度
        dif_strength = (df['DIF'] - df['DIF'].rolling(50).mean()) / df['DIF'].rolling(50).std()
        hist_strength = df['MACD_hist'] / abs(df['DIF']).rolling(20).mean()
        trend_strength = df['DIF'].diff(5) / abs(df['DIF'].diff(5)).rolling(20).mean()
        
        # 综合强度
        strength = (dif_strength.fillna(0) * 0.4 + 
                   hist_strength.fillna(0) * 0.3 + 
                   trend_strength.fillna(0) * 0.3)
        
        return strength
    
    def create_hmm_features(self):
        """创建HMM特征矩阵"""
        features = [
            'returns',
            'volatility_20',
            'rsi',
            'bb_position',
            'DIF',
            'MACD_hist',
            'macd_strength',
            'atr_ratio'
        ]
        
        # 选择可用的特征
        available_features = [f for f in features if f in self.data.columns]
        feature_data = self.data[available_features].values
        
        # 标准化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        return scaled_features, scaler
    
    def train_hmm_model(self):
        """训练隐马尔可夫模型"""
        # 准备特征数据
        features, self.hmm_scaler = self.create_hmm_features()
        
        # 训练HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_hmm_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
            tol=1e-6
        )
        
        # 拟合模型
        self.hmm_model.fit(features)
        
        # 预测市场状态
        self.data['market_state'] = self.hmm_model.predict(features)
        
        # 计算状态概率
        state_probs = self.hmm_model.predict_proba(features)
        for i in range(self.n_hmm_states):
            self.data[f'state_prob_{i}'] = state_probs[:, i]
        
        # 分析各状态特征
        self.analyze_market_states()
    
    def analyze_market_states(self):
        """分析各个市场状态的特征"""
        state_analysis = {}
        
        for state in range(self.n_hmm_states):
            state_data = self.data[self.data['market_state'] == state]
            
            analysis = {
                'count': len(state_data),
                'avg_return': state_data['returns'].mean(),
                'volatility': state_data['returns'].std(),
                'avg_macd': state_data['DIF'].mean(),
                'win_rate': (state_data['returns'] > 0).mean() if len(state_data) > 0 else 0
            }
            
            state_analysis[f'State_{state}'] = analysis
        
        self.state_analysis = state_analysis
        
        # 识别状态类型
        self.identify_state_types()
    
    def identify_state_types(self):
        """识别各个市场状态的类型"""
        state_types = {}
        
        for state, analysis in self.state_analysis.items():
            avg_return = analysis['avg_return']
            volatility = analysis['volatility']
            avg_macd = analysis['avg_macd']
            
            if avg_return > 0.001 and volatility < 0.02:
                state_type = "稳定上涨"
            elif avg_return > 0.001 and volatility >= 0.02:
                state_type = "波动上涨"
            elif avg_return < -0.001 and volatility < 0.02:
                state_type = "稳定下跌"
            elif avg_return < -0.001 and volatility >= 0.02:
                state_type = "波动下跌"
            elif abs(avg_return) <= 0.001 and volatility < 0.015:
                state_type = "横盘整理"
            else:
                state_type = "高波动震荡"
            
            state_types[state] = state_type
        
        self.state_types = state_types
    
    def generate_trading_signals(self):
        """基于HMM状态和MACD信号生成交易信号"""
        df = self.data.copy()
        
        # 初始化信号列
        df['signal'] = 0  # 0: 持有, 1: 买入, -1: 卖出
        df['position'] = 0  # 持仓方向
        df['signal_strength'] = 0.0  # 信号强度
        
        # 交易参数
        position = 0
        stop_loss_level = 0.02  # 2%止损
        take_profit_level = 0.04  # 4%止盈
        entry_prices = []
        
        for i in range(1, len(df)):
            current_state = df['market_state'].iloc[i]
            prev_state = df['market_state'].iloc[i-1]
            
            # MACD信号
            macd_signal = 0
            signal_strength = 0
            
            # 1. 基础MACD信号
            if df['golden_cross'].iloc[i]:
                macd_signal += 1
                signal_strength += 0.3
            if df['death_cross'].iloc[i]:
                macd_signal -= 1
                signal_strength += 0.3
            
            # 2. 水下信号的增强权重
            if df['underwater_golden'].iloc[i]:
                macd_signal += 1
                signal_strength += 0.4
            if df['underwater_death'].iloc[i]:
                macd_signal -= 1
                signal_strength += 0.4
            
            # 3. 双形态信号的增强权重
            if df['double_valley'].iloc[i]:
                macd_signal += 2  # 双波谷是强烈看涨信号
                signal_strength += 0.5
            if df['double_peak'].iloc[i]:
                macd_signal -= 2  # 双波峰是强烈看跌信号
                signal_strength += 0.5
            
            # 4. 背离信号
            if df['price_macd_divergence'].iloc[i] == 1:
                macd_signal += 1
                signal_strength += 0.6
            elif df['price_macd_divergence'].iloc[i] == -1:
                macd_signal -= 1
                signal_strength += 0.6
            
            # HMM状态过滤
            state_type = self.state_types.get(f'State_{current_state}', '未知')
            
            # 状态过滤规则
            state_filter = 1.0
            if state_type in ["稳定下跌", "波动下跌"]:
                # 下跌状态中，只考虑强烈的看涨信号
                if macd_signal < 0:
                    state_filter = 0.3  # 降低看跌信号权重
                elif macd_signal >= 2:  # 强烈看涨信号
                    state_filter = 1.0
                else:
                    state_filter = 0.0
            elif state_type in ["稳定上涨", "波动上涨"]:
                # 上涨状态中，只考虑强烈的看跌信号
                if macd_signal > 0:
                    state_filter = 0.3  # 降低看涨信号权重
                elif macd_signal <= -2:  # 强烈看跌信号
                    state_filter = 1.0
                else:
                    state_filter = 0.0
            elif state_type == "横盘整理":
                # 横盘状态，需要更强的信号
                if abs(macd_signal) >= 2:
                    state_filter = 1.0
                else:
                    state_filter = 0.0
            
            # 综合信号强度
            final_signal_strength = signal_strength * state_filter * df['macd_strength'].iloc[i]
            
            # 生成交易信号
            if position == 0:  # 空仓状态
                # 开多条件
                if macd_signal >= 1 and final_signal_strength > 0.5:
                    if state_type not in ["稳定下跌", "波动下跌"]:
                        df.loc[df.index[i], 'signal'] = 1
                        position = 1
                        entry_prices.append(df['close'].iloc[i])
                
                # 开空条件
                elif macd_signal <= -1 and final_signal_strength < -0.5:
                    if state_type not in ["稳定上涨", "波动上涨"]:
                        df.loc[df.index[i], 'signal'] = -1
                        position = -1
                        entry_prices.append(df['close'].iloc[i])
            
            elif position == 1:  # 持有多头
                current_price = df['close'].iloc[i]
                entry_price = entry_prices[-1]
                
                # 止损/止盈检查
                if (current_price <= entry_price * (1 - stop_loss_level)) or \
                   (current_price >= entry_price * (1 + take_profit_level)):
                    df.loc[df.index[i], 'signal'] = -1  # 平仓
                    position = 0
                    entry_prices.pop()
                
                # MACD反转信号平仓
                elif macd_signal <= -1 and final_signal_strength < -0.3:
                    df.loc[df.index[i], 'signal'] = -1
                    position = 0
                    entry_prices.pop()
            
            elif position == -1:  # 持有空头
                current_price = df['close'].iloc[i]
                entry_price = entry_prices[-1]
                
                # 止损/止盈检查
                if (current_price >= entry_price * (1 + stop_loss_level)) or \
                   (current_price <= entry_price * (1 - take_profit_level)):
                    df.loc[df.index[i], 'signal'] = 1  # 平仓
                    position = 0
                    entry_prices.pop()
                
                # MACD反转信号平仓
                elif macd_signal >= 1 and final_signal_strength > 0.3:
                    df.loc[df.index[i], 'signal'] = 1
                    position = 0
                    entry_prices.pop()
            
            df.loc[df.index[i], 'position'] = position
            df.loc[df.index[i], 'signal_strength'] = final_signal_strength
        
        self.data = df
    
    def analyze_performance(self):
        """分析策略绩效"""
        df = self.data.copy()
        
        # 计算策略收益
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()
        
        # 计算基准收益（买入持有）
        benchmark_returns = df['cumulative_returns'].iloc[-1] - 1
        strategy_returns = df['cumulative_strategy_returns'].iloc[-1] - 1
        
        # 计算风险指标
        strategy_volatility = df['strategy_returns'].std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
        
        # 最大回撤
        cumulative = df['cumulative_strategy_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        winning_trades = (df['strategy_returns'] > 0).sum()
        total_trades = (df['signal'] != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 盈亏比
        avg_win = df[df['strategy_returns'] > 0]['strategy_returns'].mean()
        avg_loss = df[df['strategy_returns'] < 0]['strategy_returns'].mean()
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        self.performance_metrics = {
            '基准收益': benchmark_returns,
            '策略收益': strategy_returns,
            '超额收益': strategy_returns - benchmark_returns,
            '年化波动率': strategy_volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '胜率': win_rate,
            '盈亏比': profit_loss_ratio,
            '总交易次数': total_trades,
            '盈利交易次数': winning_trades
        }
    
    def plot_results(self, show=True):
        """绘制策略结果并返回图表对象，便于在Streamlit中复用"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 1. 价格和持仓
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['close'], label='Price', color='black', linewidth=1)
        
        # 标记买入信号
        buy_signals = self.data[self.data['signal'] == 1]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['close'], 
                       color='green', s=100, marker='^', label='Buy Signal', zorder=5)
        
        # 标记卖出信号
        sell_signals = self.data[self.data['signal'] == -1]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['close'], 
                       color='red', s=100, marker='v', label='Sell Signal', zorder=5)
        
        ax1.set_title('Price with Trading Signals', fontsize=12)
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. MACD指标
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['DIF'], label='DIF', color='blue', linewidth=1)
        ax2.plot(self.data.index, self.data['DEA'], label='DEA', color='orange', linewidth=1)
        ax2.fill_between(self.data.index, 0, self.data['MACD_hist'], 
                        where=self.data['MACD_hist']>=0, color='green', alpha=0.3)
        ax2.fill_between(self.data.index, 0, self.data['MACD_hist'], 
                        where=self.data['MACD_hist']<0, color='red', alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('MACD Indicator', fontsize=12)
        ax2.set_ylabel('MACD')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. HMM市场状态
        ax3 = axes[2]
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_hmm_states))
        
        for state in range(self.n_hmm_states):
            state_data = self.data[self.data['market_state'] == state]
            if not state_data.empty:
                ax3.scatter(state_data.index, state_data['close'], 
                          color=colors[state], s=10, 
                          label=f'State {state}: {self.state_types.get(f"State_{state}", "")}')
        
        ax3.set_title('HMM Market States', fontsize=12)
        ax3.set_ylabel('Price')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. 累积收益对比
        ax4 = axes[3]
        ax4.plot(self.data.index, self.data['cumulative_returns'], 
                label='Buy & Hold', color='gray', linewidth=1)
        ax4.plot(self.data.index, self.data['cumulative_strategy_returns'], 
                label='MACD-HMM Strategy', color='blue', linewidth=2)
        ax4.set_title('Cumulative Returns Comparison', fontsize=12)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Returns')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()

        if show:
            plt.show()

        return fig
    
    def generate_report(self):
        """生成详细策略报告"""
        print("="*60)
        print("MACD-HMM能源期货量化策略分析报告")
        print("="*60)
        
        print("\n一、市场状态分析")
        print("-"*40)
        for state, analysis in self.state_analysis.items():
            state_type = self.state_types.get(state, "未知")
            print(f"{state} ({state_type}):")
            print(f"  出现次数: {analysis['count']}")
            print(f"  平均收益: {analysis['avg_return']:.4%}")
            print(f"  波动率: {analysis['volatility']:.4%}")
            print(f"  MACD均值: {analysis['avg_macd']:.4f}")
            print(f"  胜率: {analysis['win_rate']:.2%}")
        
        print("\n二、策略绩效指标")
        print("-"*40)
        for metric, value in self.performance_metrics.items():
            if isinstance(value, float):
                if '率' in metric or '收益' in metric or '回撤' in metric:
                    print(f"{metric}: {value:.2%}")
                elif metric == '夏普比率':
                    print(f"{metric}: {value:.2f}")
                elif metric == '盈亏比':
                    print(f"{metric}: {value:.2f}")
                else:
                    print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        print("\n三、交易统计")
        print("-"*40)
        signals = self.data['signal']
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        print(f"买入信号数量: {buy_signals}")
        print(f"卖出信号数量: {sell_signals}")
        print(f"总持仓时间比例: {(self.data['position'] != 0).sum() / len(self.data):.2%}")
        
        print("\n四、风险提示")
        print("-"*40)
        print("1. 策略基于历史数据，不代表未来表现")
        print("2. 能源期货波动较大，需严格控制仓位")
        print("3. 建议结合基本面分析和风险管理")


# 示例使用：原油期货策略
def run_crude_oil_strategy():
    """运行原油期货交易策略示例"""
    
    # 模拟原油期货数据（实际应用时替换为真实数据）
    np.random.seed(42)
    n_periods = 1000
    
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='4H')
    
    # 生成模拟价格（带有趋势和波动性）
    base_price = 60
    trend = np.cumsum(np.random.randn(n_periods) * 0.001)
    seasonal = np.sin(np.arange(n_periods) * 2 * np.pi / (252/4)) * 5
    noise = np.random.randn(n_periods) * 1.5
    
    close = base_price + trend * 10 + seasonal + noise
    high = close + np.abs(np.random.randn(n_periods) * 0.5)
    low = close - np.abs(np.random.randn(n_periods) * 0.5)
    open_price = close.shift(1) + np.random.randn(n_periods) * 0.3
    
    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(10000, 50000, n_periods)
    }, index=dates)
    
    data = data.dropna()
    
    print("正在运行MACD-HMM原油期货交易策略...")
    print(f"数据周期: {len(data)} 个4小时K线")
    print(f"时间范围: {data.index[0]} 至 {data.index[-1]}")
    
    # 创建策略实例
    strategy = EnhancedMACD_HMM_Strategy(
        data=data,
        fast_period=12,
        slow_period=26,
        signal_period=9,
        n_hmm_states=5,  # 原油市场通常有5种状态
        n_hmm_features=8
    )
    
    # 生成报告
    strategy.generate_report()
    
    # 绘制结果
    strategy.plot_results()
    
    return strategy


def fetch_yfinance_data(symbol='CL=F', period='2y', interval='4h'):
    """使用yfinance获取能源品种4小时数据"""
    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)

    if raw.empty:
        raise ValueError(f"无法从yfinance获取{symbol}的数据，请检查代码或网络连接")

    raw = raw.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    })

    return raw[['open', 'high', 'low', 'close', 'volume']].dropna()


def run_yfinance_energy_strategy(symbol='CL=F', period='2y', interval='4h'):
    """基于yfinance 4小时数据的能源品种MACD-HMM策略示例"""
    print(f"从yfinance下载{symbol}的{period}、{interval}数据...")
    data = fetch_yfinance_data(symbol=symbol, period=period, interval=interval)
    print(f"获取到{len(data)}根K线，时间范围: {data.index[0]} 至 {data.index[-1]}")

    strategy = EnhancedMACD_HMM_Strategy(
        data=data,
        fast_period=12,
        slow_period=26,
        signal_period=9,
        n_hmm_states=5,
        n_hmm_features=8
    )

    strategy.generate_report()
    return strategy


@st.cache_data(show_spinner=False)
def cached_fetch_yfinance_data(symbol: str, period: str, interval: str):
    """缓存yfinance请求，避免Streamlit重复运行时频繁下载"""
    return fetch_yfinance_data(symbol=symbol, period=period, interval=interval)


def render_streamlit_app():
    """简易的Streamlit界面，方便在Web端运行策略"""
    st.set_page_config(page_title="MACD-HMM能源策略", layout="wide")
    st.title("MACD-HMM能源期货量化策略（4小时级别）")
    st.markdown(
        "结合**MACD双波峰/波谷 + 金叉/死叉**与**HMM市场状态**过滤的多因子策略，"
        "可直接拉取 yfinance 的能源合约4小时K线。"
    )

    with st.sidebar:
        st.header("参数设置")
        symbol = st.text_input("yfinance代码", value="CL=F", help="例如：原油 CL=F，天然气 NG=F")
        period = st.selectbox("历史区间", options=["6mo", "1y", "2y", "5y"], index=2)
        interval = st.selectbox("K线周期", options=["1h", "4h", "1d"], index=1)
        fast_period = st.slider("MACD 快线", min_value=6, max_value=20, value=12, step=1)
        slow_period = st.slider("MACD 慢线", min_value=20, max_value=40, value=26, step=1)
        signal_period = st.slider("MACD 信号线", min_value=5, max_value=15, value=9, step=1)
        n_hmm_states = st.slider("HMM状态数", min_value=3, max_value=8, value=5, step=1)

        run_clicked = st.button("运行策略", type="primary")

    if not run_clicked:
        st.info("在左侧设置参数后点击 **运行策略** 开始计算。")
        return

    with st.spinner("正在下载数据并运行策略，请稍候..."):
        try:
            data = cached_fetch_yfinance_data(symbol=symbol, period=period, interval=interval)
        except Exception as exc:  # pragma: no cover - 仅用于交互式提示
            st.error(f"下载 {symbol} 数据失败: {exc}")
            return

        strategy = EnhancedMACD_HMM_Strategy(
            data=data,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            n_hmm_states=n_hmm_states,
            n_hmm_features=8
        )

    st.success("计算完成 ✅")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("策略绩效")
        metrics_df = pd.DataFrame(strategy.performance_metrics, index=["Value"]).T
        st.dataframe(metrics_df)

    with col2:
        st.subheader("市场状态分类")
        state_df = pd.DataFrame(strategy.state_analysis).T
        state_df["state_type"] = state_df.index.map(lambda idx: strategy.state_types.get(idx, "未知"))
        st.dataframe(state_df)

    st.subheader("MACD事件日志（最近50条）")
    st.dataframe(strategy.build_macd_event_log().tail(50))

    st.subheader("图表")
    fig = strategy.plot_results(show=False)
    st.pyplot(fig, use_container_width=True)


# 运行策略
if __name__ == "__main__":
    render_streamlit_app()
