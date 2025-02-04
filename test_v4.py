import pandas as pd
import numpy as np
from collections import Counter
import requests
import warnings
import os

from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def update_lotto_csv(csv_file='lotto_number.csv', max_consecutive_fails=5):
    try:
        existing_df = pd.read_csv(csv_file)
        last_round = existing_df['회차'].max()
        start_round = last_round + 1
        print(f"[update_lotto_csv] 기존 데이터의 마지막 회차: {last_round}")
        print(f"  => {start_round}회차부터 데이터를 추가로 수집합니다...")
    except FileNotFoundError:
        existing_df = pd.DataFrame()
        start_round = 1
        print("[update_lotto_csv] 기존 데이터 없음: 1회차부터 수집 시작...")

    url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="
    new_data = []
    i = start_round
    consecutive_fails = 0

    while True:
        if consecutive_fails >= max_consecutive_fails:
            print(f"[update_lotto_csv] 연속 {max_consecutive_fails}회 fail. 더 이상 데이터를 가져올 수 없어 중단합니다.")
            break

        try:
            response = requests.get(url + str(i), timeout=5)
            lotto_data = response.json()
            if lotto_data.get('returnValue') == 'fail':
                consecutive_fails += 1
                i += 1
                continue
            else:
                consecutive_fails = 0

            numbers = [
                lotto_data['drwtNo1'],
                lotto_data['drwtNo2'],
                lotto_data['drwtNo3'],
                lotto_data['drwtNo4'],
                lotto_data['drwtNo5'],
                lotto_data['drwtNo6']
            ]
            new_data.append({
                '회차': i,
                '번호1': numbers[0],
                '번호2': numbers[1],
                '번호3': numbers[2],
                '번호4': numbers[3],
                '번호5': numbers[4],
                '번호6': numbers[5]
            })

            if i % 10 == 0:
                print(f"  -> {i}회차 데이터 수집 완료")
            i += 1

        except Exception as e:
            print(f"[update_lotto_csv] {i}회차 데이터 수집 중 오류: {e}")
            consecutive_fails += 1
            i += 1

    if new_data:
        new_df = pd.DataFrame(new_data)
        df = pd.concat([existing_df, new_df], ignore_index=True)
        df.drop_duplicates(subset=['회차'], inplace=True)
        df.sort_values(by='회차', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"[update_lotto_csv] 새로운 데이터 {len(new_data)}개(회차) 추가 완료")
    else:
        df = existing_df
        print("[update_lotto_csv] 추가할 새로운 데이터가 없습니다.")

    return df


def find_suitable_apriori_thresholds(transaction_df,
                                     support_candidates=[0.005, 0.004, 0.003, 0.002],
                                     conf_candidates=[0.1, 0.08, 0.05],
                                     max_rules_allowed=300):
    transaction_bool = transaction_df.astype(bool)
    best_support = None
    best_conf = None
    found_rules = None

    support_candidates_sorted = sorted(support_candidates, reverse=True)
    conf_candidates_sorted = sorted(conf_candidates, reverse=True)

    print("[find_suitable_apriori_thresholds] 동적 탐색:")
    for sup in support_candidates_sorted:
        freq_itemsets = apriori(transaction_bool, min_support=sup, use_colnames=True)
        if len(freq_itemsets) == 0:
            print(f"  - support={sup} => 빈번항목셋 0개 -> continue")
            continue

        print(f"  - support={sup}, 빈번항목셋 {len(freq_itemsets)}개")

        for conf in conf_candidates_sorted:
            # 구버전 호환을 위해 num_itemsets 추가
            rules = association_rules(
                freq_itemsets,
                metric="confidence",
                min_threshold=conf,
                num_itemsets=len(freq_itemsets)  # ← 여기 추가
            )
            n_rules = len(rules)

            if n_rules == 0:
                print(f"     -> conf={conf} => 규칙 0개 -> continue")
                continue

            if n_rules > max_rules_allowed:
                print(f"     -> conf={conf} => 규칙 {n_rules}개 (>{max_rules_allowed}개) -> continue")
                continue

            print(f"     -> conf={conf} => 규칙 {n_rules}개 (적합), 바로 사용.")
            best_support = sup
            best_conf = conf
            found_rules = rules
            return best_support, best_conf, found_rules

        print(f"  => support={sup}에서 모든 conf 후보 실패. 다음 support로...")

    # fallback
    sup = support_candidates[-1]
    conf = conf_candidates[-1]
    print(f"[find_suitable_apriori_thresholds] 모든 후보 실패 => fallback => sup={sup}, conf={conf}")

    freq_itemsets = apriori(transaction_bool, min_support=sup, use_colnames=True)
    if len(freq_itemsets) == 0:
        print("  => fallback에서도 빈번항목셋 0 => 규칙 사용 불가")
        return None, None, None

    rules = association_rules(
        freq_itemsets,
        metric="confidence",
        min_threshold=conf,
        num_itemsets=len(freq_itemsets)  # 구버전 호환
    )
    n_rules = len(rules)
    if n_rules == 0:
        print("  => fallback에서 규칙 0개 => 사용 불가")
        return None, None, None
    if n_rules > max_rules_allowed:
        print(f"  => fallback 규칙 {n_rules}개 => {max_rules_allowed} 초과 => 사용 불가")
        return None, None, None

    print(f"  => fallback 사용 => 규칙 {n_rules}개")
    return sup, conf, rules


class AdvancedLottoNumberGenerator:
    def __init__(self, df, recent_weight=0.8, config=None):
        self.df = df.copy()
        self.numbers_range = range(1, 46)
        self.recent_weight = recent_weight

        if config is None:
            config = {}

        self.config = {
            'support_candidates': config.get('support_candidates', [0.005, 0.004, 0.003, 0.002]),
            'conf_candidates': config.get('conf_candidates', [0.1, 0.08, 0.05]),
            'max_rules_allowed': config.get('max_rules_allowed', 300),

            'max_consecutive_pairs': config.get('max_consecutive_pairs', 2),
            'max_same_lastdigit_ratio': config.get('max_same_lastdigit_ratio', 0.5),
            'max_interval_ratio': config.get('max_interval_ratio', 0.5),

            'sum_range_mode': config.get('sum_range_mode', 'auto'),
            'manual_sum_min': config.get('manual_sum_min', 110),
            'manual_sum_max': config.get('manual_sum_max', 180),
            'sum_std_multiplier': config.get('sum_std_multiplier', 1.5),
        }

        self.df.sort_values(by='회차', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.total_rows = len(self.df)

        self.frequency_result = None
        self.patterns = None
        self.pattern_prob = None
        self.correlations_result = None
        self.triple_correlations = None

        self.association_rules_df = pd.DataFrame()
        self.final_support = None
        self.final_confidence = None

        self.weight_corr = 0.4
        self.weight_markov = 0.4
        self.weight_assoc = 0.2

    def analyze_frequency(self):
        print("[-] 지수 가중 빈도분석...")
        all_numbers = []
        weights = []
        total_rows = self.total_rows

        for idx, row in self.df.iterrows():
            ratio = (idx + 1) / total_rows
            time_weight = np.exp(ratio * self.recent_weight)
            nums = [row[f'번호{i}'] for i in range(1, 7)]
            all_numbers.extend(nums)
            weights.extend([time_weight]*6)

        freq = {}
        for num, w in zip(all_numbers, weights):
            freq[num] = freq.get(num, 0) + w

        self.frequency_result = freq
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top10 = [f"{n}({val:.2f})" for (n, val) in sorted_freq[:10]]
        print(f"  => 상위 10개 번호: {top10}")

    def analyze_patterns(self):
        print("[-] 패턴 분석...")
        patterns = {
            'sum_range': [],
            'odd_even_patterns': [],
            'consecutive_counts': [],
        }

        for _, row in self.df.iterrows():
            numbers = sorted([row[f'번호{i}'] for i in range(1, 7)])
            patterns['sum_range'].append(sum(numbers))
            odd_even = ''.join(['O' if (n % 2) else 'E' for n in numbers])
            patterns['odd_even_patterns'].append(odd_even)
            consecutive_count = sum(1 for i in range(len(numbers)-1)
                                    if numbers[i+1] - numbers[i] == 1)
            patterns['consecutive_counts'].append(consecutive_count)

        self.patterns = patterns
        sr = patterns['sum_range']
        print(f"  => 과거 당첨번호 합계: min={min(sr)}, max={max(sr)}")

    def calculate_pattern_probabilities(self):
        print("[-] 패턴 확률 계산...")
        if self.patterns is None:
            print("  * 패턴 데이터가 없어 중단.")
            return

        prob = {}
        oe_counter = Counter(self.patterns['odd_even_patterns'])
        total_oe = len(self.patterns['odd_even_patterns'])
        prob['odd_even'] = {k: v/total_oe for k, v in oe_counter.items()}

        cc_counter = Counter(self.patterns['consecutive_counts'])
        total_cc = len(self.patterns['consecutive_counts'])
        prob['consecutive'] = {k: v/total_cc for k, v in cc_counter.items()}

        self.pattern_prob = prob

        sorted_oe = sorted(prob['odd_even'].items(), key=lambda x: x[1], reverse=True)[:5]
        print("  => 자주 나온 홀짝 패턴 TOP5:", sorted_oe)

    def analyze_correlations(self):
        print("[-] 번호 상관관계 분석...")
        correlations = {}
        triple_correlations = {}
        for i in self.numbers_range:
            correlations[i] = {}
            triple_correlations[i] = {}
            for j in self.numbers_range:
                correlations[i][j] = 0.0
                triple_correlations[i][j] = {}
                for k in self.numbers_range:
                    triple_correlations[i][j][k] = 0.0

        total_rows = self.total_rows
        for idx, row in self.df.iterrows():
            weight = 0.8 + 0.4 * (idx / total_rows)
            nums = sorted([row[f'번호{i}'] for i in range(1, 7)])

            for a_i in range(len(nums)):
                for b_i in range(a_i+1, len(nums)):
                    a, b = nums[a_i], nums[b_i]
                    correlations[a][b] += weight
                    correlations[b][a] += weight
                    for c_i in range(b_i+1, len(nums)):
                        c = nums[c_i]
                        triple_correlations[a][b][c] += weight
                        triple_correlations[a][c][b] += weight
                        triple_correlations[b][a][c] += weight
                        triple_correlations[b][c][a] += weight
                        triple_correlations[c][a][b] += weight
                        triple_correlations[c][b][a] += weight

        for i in self.numbers_range:
            for j in self.numbers_range:
                if i != j:
                    correlations[i][j] /= total_rows
                    for k in self.numbers_range:
                        if k not in (i, j):
                            triple_correlations[i][j][k] /= total_rows

        self.correlations_result = correlations
        self.triple_correlations = triple_correlations
        print("  => 상관계수 계산 완료.")

    def perform_association_rule_mining(self):
        print("[-] Apriori 연관규칙 분석...")
        transaction_df = pd.DataFrame(0, index=self.df.index, columns=self.numbers_range)
        for idx, row in self.df.iterrows():
            nums = [row[f'번호{i}'] for i in range(1, 7)]
            for n in nums:
                transaction_df.loc[idx, n] = 1

        sup, conf, rules = find_suitable_apriori_thresholds(
            transaction_df,
            support_candidates=self.config['support_candidates'],
            conf_candidates=self.config['conf_candidates'],
            max_rules_allowed=self.config['max_rules_allowed']
        )

        if (sup is None) or (conf is None) or (rules is None) or rules.empty:
            print("  => 적절한 규칙 찾지 못함. 빈 DataFrame 유지")
            self.final_support = None
            self.final_confidence = None
            self.association_rules_df = pd.DataFrame()
            return

        self.final_support = sup
        self.final_confidence = conf
        # 2차 필터링 예시
        filtered_rules = rules[
            (rules['lift'] > 1.0) &
            (rules['support'] > 0.001) &
            (rules['support'] < 0.3)
        ]
        if len(filtered_rules) == 0:
            print("  => 2차 필터링 후 규칙이 없음. 빈 DataFrame 유지")
            self.association_rules_df = pd.DataFrame()
        else:
            self.association_rules_df = filtered_rules.reset_index(drop=True)

        print(f"  => 최종 규칙 {len(self.association_rules_df)}개: min_support={sup}, min_confidence={conf}")

    def preprocess_all(self):
        self.analyze_frequency()
        self.analyze_patterns()
        self.calculate_pattern_probabilities()
        self.analyze_correlations()
        self.perform_association_rule_mining()

    def generate_one_combination(self, max_tries=100):
        if self.association_rules_df.empty:
            filtered_rules = []
        else:
            filtered_rules = []
            for _, row in self.association_rules_df.iterrows():
                ant = tuple(sorted(row['antecedents']))
                con = tuple(sorted(row['consequents']))
                sup = row['support']
                cf = row['confidence']
                lf = row['lift']
                filtered_rules.append((ant, con, sup, cf, lf))

        available = list(self.numbers_range)
        numbers = []

        if not self.frequency_result:
            freq = {n: 1 for n in available}
        else:
            freq = self.frequency_result
        freq_vals = np.array([freq.get(n, 0) for n in available])
        if freq_vals.sum() == 0:
            p_init = np.ones_like(freq_vals)/len(freq_vals)
        else:
            p_init = freq_vals / freq_vals.sum()

        first_num = np.random.choice(available, p=p_init)
        numbers.append(first_num)
        available.remove(first_num)

        tries = 0
        while len(numbers) < 6 and tries < max_tries:
            tries += 1
            weights = []
            for num in available:
                base_weight = sum(self.correlations_result[n].get(num, 0) for n in numbers)

                markov_weight = 0
                if len(numbers) >= 2:
                    last_two = numbers[-2:]
                    markov_weight = self.triple_correlations[last_two[0]][last_two[1]].get(num, 0)

                assoc_weight = 0
                for (ant, con, sup, cf, lf) in filtered_rules:
                    if set(ant).issubset(set(numbers)):
                        if num in con:
                            assoc_weight += (lf - 1) * sup * cf

                final_weight = (self.weight_corr * base_weight) + \
                               (self.weight_markov * markov_weight) + \
                               (self.weight_assoc * assoc_weight)
                weights.append(final_weight)

            w_arr = np.array(weights)
            if w_arr.sum() == 0:
                p = np.ones_like(w_arr) / len(w_arr)
            else:
                p = w_arr / w_arr.sum()

            candidate = np.random.choice(available, p=p)
            temp = numbers + [candidate]
            if self.validate_partial_combination(temp):
                numbers.append(candidate)
                available.remove(candidate)
            else:
                # 그냥 재시도
                pass

        if len(numbers) < 6:
            return None

        numbers_sorted = sorted(numbers)
        if self.validate_combination(numbers_sorted):
            return numbers_sorted
        else:
            return None

    def validate_partial_combination(self, nums):
        sorted_nums = sorted(nums)
        ccount = sum(1 for i in range(len(sorted_nums)-1)
                     if sorted_nums[i+1]-sorted_nums[i] == 1)
        if ccount > self.config['max_consecutive_pairs']:
            return False

        last_digits = [n % 10 for n in nums]
        cnt_max_digit = max(Counter(last_digits).values())
        if cnt_max_digit > len(nums) * self.config['max_same_lastdigit_ratio']:
            return False

        intervals = [1 + (n-1)//10 for n in nums]
        cnt_interval = max(Counter(intervals).values())
        if cnt_interval > len(nums) * self.config['max_interval_ratio']:
            return False

        return True

    def validate_combination(self, nums):
        if not self.validate_partial_combination(nums):
            return False

        if self.pattern_prob:
            oe = ''.join(['O' if n % 2 else 'E' for n in nums])
            if oe not in self.pattern_prob['odd_even']:
                return False

        cc = sum(1 for i in range(5) if nums[i+1]-nums[i] == 1)
        if self.pattern_prob and cc not in self.pattern_prob['consecutive']:
            return False

        s = sum(nums)
        if self.config['sum_range_mode'] == 'auto' and self.patterns is not None:
            sums = self.patterns['sum_range']
            mean_sum = np.mean(sums)
            std_sum = np.std(sums)
            auto_min = mean_sum - self.config['sum_std_multiplier'] * std_sum
            auto_max = mean_sum + self.config['sum_std_multiplier'] * std_sum
            if not (auto_min <= s <= auto_max):
                return False
        elif self.config['sum_range_mode'] == 'manual':
            if not (self.config['manual_sum_min'] <= s <= self.config['manual_sum_max']):
                return False

        return True

    def generate_numbers(self, n_combinations=5, max_attempts=5000):
        print(f"\n[번호 생성] 요청 조합 수={n_combinations}")
        if self.frequency_result is None:
            self.preprocess_all()

        results = []
        attempts = 0
        while len(results) < n_combinations and attempts < max_attempts:
            attempts += 1
            combo = self.generate_one_combination(max_tries=50)
            if combo is not None:
                if combo not in results:
                    results.append(combo)

        if len(results) < n_combinations:
            print(f"  => {n_combinations}개 중 {len(results)}개만 생성")

        return results


if __name__ == "__main__":
    csv_file = 'lotto_number.csv'
    df = update_lotto_csv(csv_file=csv_file)

    print("\n[데이터프레임 정보]")
    print(df.info())
    print(df.tail())

    config = {
        'support_candidates': [0.01, 0.009, 0.008, 0.007, 0.005],
        'conf_candidates': [0.2, 0.15, 0.1],
        'max_rules_allowed': 300,

        'max_consecutive_pairs': 2,
        'max_same_lastdigit_ratio': 0.5,
        'max_interval_ratio': 0.5,

        'sum_range_mode': 'auto',
        'sum_std_multiplier': 1.5,
    }

    generator = AdvancedLottoNumberGenerator(df, recent_weight=0.8, config=config)
    recommended = generator.generate_numbers(n_combinations=5)

    print("\n[최종 추천 번호 조합]")
    for i, combo in enumerate(recommended, start=1):
        print(f"  {i} : {combo}")

    print("\n--- 완료 ---")
