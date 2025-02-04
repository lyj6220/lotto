from flask import Flask, render_template, request, jsonify
from test_v4 import update_lotto_csv, AdvancedLottoNumberGenerator
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from models import db, Category, Item, Vote, VotingSession

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///votes.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# 네이버 클라우드의 임시 저장소 사용
CSV_PATH = '/tmp/lotto_number.csv'

# 데이터베이스 생성 확인
def init_db():
    if not os.path.exists('instance/votes.db'):
        with app.app_context():
            db.create_all()

# 앱 시작 시 DB 초기화 확인
init_db()

# 상단에 추가
ADMIN_PASSWORD = "0000"  # 원하는 비밀번호로 변경하세요

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        n_combinations = int(request.form.get('n_combinations', 5))
        
        # CSV 파일 업데이트
        df = update_lotto_csv()
        
        # 설정값
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
        
        # 번호 생성기 초기화 및 전처리
        generator = AdvancedLottoNumberGenerator(df, recent_weight=0.8, config=config)
        generator.preprocess_all()
        
        # 번호 생성
        recommended = generator.generate_numbers(n_combinations=n_combinations)
        
        # numpy int64를 Python int로 변환
        recommended = [[int(num) for num in combo] for combo in recommended]
        
        # 분석 결과 저장을 위한 딕셔너리
        analysis_results = {
            'frequency': [],
            'patterns': {
                'odd_even': [],
                'consecutive': []
            }
        }
        
        # 빈도 분석 결과
        if generator.frequency_result:
            sorted_freq = sorted(generator.frequency_result.items(), key=lambda x: x[1], reverse=True)
            analysis_results['frequency'] = [
                {'number': int(n), 'weight': f"{float(w):.2f}"} 
                for n, w in sorted_freq[:10]
            ]
        
        # 패턴 분석 결과
        if generator.pattern_prob:
            analysis_results['patterns']['odd_even'] = [
                [pattern, float(prob)] 
                for pattern, prob in sorted(
                    generator.pattern_prob['odd_even'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            ]
            analysis_results['patterns']['consecutive'] = [
                [int(count), float(prob)]
                for count, prob in sorted(
                    generator.pattern_prob['consecutive'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
            ]
        
        return jsonify({
            'success': True,
            'combinations': recommended,
            'analysis': analysis_results
        })
        
    except Exception as e:
        print(f"Error in generate: {str(e)}")  # 서버 콘솔에 오류 출력
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/vote', methods=['POST'])
def vote():
    try:
        data = request.json
        item_id = data.get('item_id')
        user_ip = request.remote_addr
        
        new_vote = Vote(item_id=item_id, user_ip=user_ip)
        db.session.add(new_vote)
        db.session.commit()
        
        return jsonify({'message': '투표가 완료되었습니다'})
    except Exception as e:
        db.session.rollback()  # 예외 발생 시 롤백
        print(f"Error in vote: {str(e)}")
        return jsonify({'error': '투표 처리 중 오류가 발생했습니다'}), 500
    finally:
        db.session.close()  # 세션 정리

@app.route('/vote/results')
def get_results():
    # Top 3 항목 조회
    top_items = db.session.query(
        Item.name, 
        db.func.count(Vote.id).label('vote_count')
    ).join(Vote).group_by(Item.id).order_by(db.desc('vote_count')).limit(3).all()
    
    # Row 객체를 리스트로 변환
    results = [{'name': item[0], 'votes': item[1]} for item in top_items]
    
    return jsonify({'top_items': results})

@app.route('/admin/reset', methods=['POST'])
def reset_votes():
    try:
        data = request.json
        password = data.get('password')
        
        if password != ADMIN_PASSWORD:
            return jsonify({'error': '관리자 인증 실패'}), 401
            
        Vote.query.delete()
        db.session.commit()
        return jsonify({'message': '투표가 초기화되었습니다'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/new-session', methods=['POST'])
def new_voting_session():
    # 실제 구현시에는 관리자 인증 필요
    current_session = VotingSession.query.filter_by(is_active=True).first()
    if current_session:
        current_session.is_active = False
        
    new_session = VotingSession(
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow() + timedelta(days=1),
        is_active=True
    )
    db.session.add(new_session)
    db.session.commit()
    return jsonify({'message': '새로운 투표가 시작되었습니다'})

@app.route('/categories')
def get_categories():
    try:
        categories = Category.query.filter_by(parent_id=None).all()
        result = []
        for cat in categories:
            subcats = Category.query.filter_by(parent_id=cat.id).all()
            cat_data = {
                'id': cat.id,
                'name': cat.name,
                'subcategories': [{
                    'id': subcat.id,
                    'name': subcat.name,
                    'items': [{
                        'id': item.id,
                        'name': item.name,
                        'votes': len(item.votes)
                    } for item in Item.query.filter_by(category_id=subcat.id).all()]
                } for subcat in subcats]
            }
            result.append(cat_data)
        return jsonify(result)
    except Exception as e:
        print(f"Error in get_categories: {str(e)}")  # 서버 콘솔에 오류 출력
        return jsonify({'error': str(e)}), 500

# Cloud Functions용 핸들러 추가
def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    app.run() 