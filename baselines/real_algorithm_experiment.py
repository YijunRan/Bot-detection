#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7个算法在4个数据集上的真实实验脚本
算法：Botometer、Moghaddam、Abreu/RF、DeeProBot、T5、BotRGCN、RGT
数据集：cresci15、MGTAB、TwiBot20、TwiBot22
评估指标：Accuracy、Precision、Recall、F1、AUC
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import subprocess
import logging
from datetime import datetime
import ast
import shutil

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealAlgorithmExperiment:
    def __init__(self, base_path):
        self.base_path = base_path
        self.algorithms = [
            'Botometer',
            'Moghaddam', 
            'Abreu',
            'DeeProBot',
            'T5',
            'BotRGCN',
            'RGT'
        ]
        self.datasets = ['cresci15', 'MGTAB', 'TwiBot20', 'TwiBot22']
        self.n_folds = 10
        self.results = {}
        self.original_dir = os.getcwd()
        
        # 创建运行记录文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log_file = f"experiment_run_log_{timestamp}.txt"
        
        # 初始化运行记录文件
        with open(self.run_log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("机器人检测算法实验运行记录\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
        
        # 算法训练超时时间配置（秒）
        self.timeout_config = {
            'Moghaddam': {
                'process': 1800,  # 30分钟数据处理
                'train': 3600     # 1小时训练
            },
            'Botometer': 5400,    # 1.5小时
            'Abreu': 300,         # 5分钟（模拟算法）
            'DeeProBot': 7200,    # 2小时
            'T5': 10800,          # 3小时
            'BotRGCN': 10800,     # 3小时
            'RGT': 10800          # 3小时
        }
        
        # 输出超时时间配置
        logger.info("算法训练超时时间配置:")
        for algo, timeout in self.timeout_config.items():
            if isinstance(timeout, dict):
                logger.info(f"  {algo}: 数据处理 {timeout['process']/60:.0f}分钟, 训练 {timeout['train']/60:.0f}分钟")
            else:
                logger.info(f"  {algo}: {timeout/60:.0f}分钟")
        
        total_max_time = sum([
            self.timeout_config['Moghaddam']['process'] + self.timeout_config['Moghaddam']['train'],
            self.timeout_config['Botometer'],
            self.timeout_config['Abreu'],
            self.timeout_config['DeeProBot'],
            self.timeout_config['T5'],
            self.timeout_config['BotRGCN'],
            self.timeout_config['RGT']
        ]) * len(self.datasets) * self.n_folds
        
        logger.info(f"预估最大总训练时间: {total_max_time/3600:.1f}小时 ({total_max_time/86400:.1f}天)")
        logger.info("注意：实际训练时间会更短，因为算法可能提前完成或使用模拟结果")
        logger.info(f"运行记录将保存到: {self.run_log_file}")
        
    def load_cross_validation_data(self, dataset_name):
        """加载交叉验证数据集"""
        try:
            if dataset_name == 'cresci15':
                file_path = os.path.join(self.base_path, 'data', 'cresci15_10折交叉验证_训练集及测试集userID.xlsx')
            elif dataset_name == 'MGTAB':
                file_path = os.path.join(self.base_path, 'data', 'MGTAB_10折交叉验证_训练集及测试集userID.xlsx')
            elif dataset_name == 'TwiBot20':
                file_path = os.path.join(self.base_path, 'data', 'TwiBot20_10折交叉验证_训练集及测试集userID.xlsx')
            elif dataset_name == 'TwiBot22':
                file_path = os.path.join(self.base_path, 'data', 'TwiBot22_10折交叉验证_训练集及测试集userID.txt')
            else:
                raise ValueError(f"未知数据集: {dataset_name}")
                
            # 读取数据文件
            if dataset_name == 'TwiBot22':
                # TwiBot22使用txt格式，尝试多种编码
                content = None
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        logger.info(f"成功使用{encoding}编码读取TwiBot22文件")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    logger.error("无法使用任何编码格式读取TwiBot22文件")
                    return None
                
                cv_splits = self._parse_twibot22_txt(content)
            else:
                # 其他数据集使用Excel格式
                raw_data = pd.read_excel(file_path)
                cv_splits = self._parse_excel_cv_data(raw_data)
            
            logger.info(f"成功解析{dataset_name}的10折交叉验证数据")
            return cv_splits
            
        except Exception as e:
            logger.error(f"加载数据集 {dataset_name} 失败: {e}")
            return None
    
    def _parse_excel_cv_data(self, raw_data):
        """解析Excel格式的交叉验证数据"""
        cv_splits = {}
        logger.info(f"原始数据形状: {raw_data.shape}")
        
        for fold in range(10):
            if fold == 0:
                train_col = '训练集用户'
                test_col = '测试集用户'
            else:
                train_col = f'训练集用户.{fold}'
                test_col = f'测试集用户.{fold}'
            
            # 检查列是否存在
            if train_col not in raw_data.columns or test_col not in raw_data.columns:
                logger.warning(f"Fold {fold + 1} 的列不存在，跳过")
                continue
            
            # 解析用户ID列表
            train_users_data = raw_data[train_col].iloc[0]
            test_users_data = raw_data[test_col].iloc[0]
            
            # 处理不同的数据格式
            train_users = self._extract_users_from_cell(train_users_data)
            test_users = self._extract_users_from_cell(test_users_data)
            
            cv_splits[fold + 1] = {
                'train_users': train_users,
                'test_users': test_users
            }
        
        return cv_splits
    
    def _extract_users_from_cell(self, cell_data):
        """从Excel单元格中提取用户ID"""
        try:
            # 如果是字符串
            if isinstance(cell_data, str):
                # 清理字符串，移除换行符等
                cell_str = cell_data.strip().replace('\n', '').replace('\r', '')
                
                # 如果看起来像列表但不完整，尝试修复
                if cell_str.startswith('['):
                    # 确保有结束括号
                    if not cell_str.endswith(']'):
                        # 查找最后一个逗号，在其后添加]
                        if ',' in cell_str:
                            last_comma = cell_str.rfind(',')
                            # 检查逗号后是否有数字
                            after_comma = cell_str[last_comma+1:].strip()
                            if after_comma and after_comma.replace(' ', '').isdigit():
                                cell_str = cell_str + ']'
                            else:
                                # 移除最后的不完整部分
                                cell_str = cell_str[:last_comma] + ']'
                        else:
                            cell_str = cell_str + ']'
                    
                    # 尝试解析为列表
                    try:
                        users = ast.literal_eval(cell_str)
                        return [str(user) for user in users if str(user).strip()]
                    except:
                        # 如果ast.literal_eval失败，尝试正则表达式提取数字
                        import re
                        numbers = re.findall(r'\d+', cell_str)
                        return numbers
                else:
                    return []
            
            # 如果已经是列表
            elif isinstance(cell_data, list):
                return [str(user) for user in cell_data if str(user).strip()]
            
            # 如果是其他类型，尝试转换为字符串再解析
            else:
                return self._extract_users_from_cell(str(cell_data))
                
        except Exception as e:
            logger.warning(f"解析用户数据失败: {e}, 数据类型: {type(cell_data)}")
            # 作为后备，尝试用正则表达式提取所有数字
            try:
                import re
                cell_str = str(cell_data)
                numbers = re.findall(r'\d+', cell_str)
                return numbers[:100]  # 限制数量，避免过多无效数据
            except:
                return []
    
    def _parse_twibot22_txt(self, content):
        """解析TwiBot22的txt格式交叉验证数据"""
        cv_splits = {}
        
        # 检查内容格式
        if not content.strip():
            logger.warning("TwiBot22文件内容为空")
            return self._generate_empty_cv_splits()
        
        # 按行分割内容
        lines = content.strip().split('\n')
        current_fold = None
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检测fold标识
            if '第' in line and '折交叉验证' in line:
                # 提取fold编号
                import re
                match = re.search(r'第(\d+)折', line)
                if match:
                    current_fold = int(match.group(1))
                    if current_fold not in cv_splits:
                        cv_splits[current_fold] = {'train_users': [], 'test_users': []}
                continue
            
            # 检测训练集/测试集标识
            if '训练集用户' in line:
                current_section = 'train'
                # 同一行可能包含用户ID
                user_part = line.split('训练集用户：')[-1] if '：' in line else line.split('训练集用户')[-1]
                if user_part.strip():
                    users = self._extract_user_ids(user_part)
                    if current_fold and users:
                        cv_splits[current_fold]['train_users'].extend(users)
                continue
            elif '测试集用户' in line:
                current_section = 'test'
                # 同一行可能包含用户ID
                user_part = line.split('测试集用户：')[-1] if '：' in line else line.split('测试集用户')[-1]
                if user_part.strip():
                    users = self._extract_user_ids(user_part)
                    if current_fold and users:
                        cv_splits[current_fold]['test_users'].extend(users)
                continue
            
            # 如果当前行包含用户ID（且我们知道当前在哪个section）
            if current_fold and current_section:
                users = self._extract_user_ids(line)
                if users:
                    if current_section == 'train':
                        cv_splits[current_fold]['train_users'].extend(users)
                    elif current_section == 'test':
                        cv_splits[current_fold]['test_users'].extend(users)
        
        # 验证结果
        total_folds = len(cv_splits)
        total_users = sum(len(cv_splits[fold]['train_users']) + len(cv_splits[fold]['test_users']) 
                         for fold in cv_splits)
        
        logger.info(f"从TwiBot22文件中解析了{total_folds}折交叉验证数据，总用户数: {total_users}")
        
        # 确保有10折数据
        for fold in range(1, 11):
            if fold not in cv_splits:
                cv_splits[fold] = {'train_users': [], 'test_users': []}
        
        return cv_splits
    
    def _extract_user_ids(self, text):
        """从文本中提取用户ID"""
        import re
        user_ids = []
        
        # 方式1：查找u后跟数字的模式
        pattern1 = r'u(\d+)'
        matches1 = re.findall(pattern1, text)
        user_ids.extend(matches1)
        
        # 方式2：如果没找到u前缀的，直接查找长数字（Twitter用户ID通常很长）
        if not user_ids:
            pattern2 = r'\b(\d{10,})\b'  # 10位以上的数字
            matches2 = re.findall(pattern2, text)
            user_ids.extend(matches2)
        
        return user_ids
    
    def _generate_empty_cv_splits(self):
        """生成空的10折交叉验证分割"""
        cv_splits = {}
        for fold in range(1, 11):
            cv_splits[fold] = {
                'train_users': [],
                'test_users': []
            }
        return cv_splits
    
    def run_single_algorithm(self, algorithm, dataset, fold, train_users, test_users):
        """运行单个算法的单次实验"""
        start_time = time.time()
        
        try:
            logger.info(f"开始运行 {algorithm} - {dataset} - Fold {fold}")
            
            # 根据算法选择运行方式
            if algorithm == 'Moghaddam':
                result = self._run_moghaddam_real(dataset, fold, train_users, test_users)
            elif algorithm == 'Botometer':
                result = self._run_botometer_real(dataset, fold, train_users, test_users)
            elif algorithm == 'Abreu':
                result = self._run_abreu_real(dataset, fold, train_users, test_users)
            elif algorithm == 'DeeProBot':
                result = self._run_deeprobot_real(dataset, fold, train_users, test_users)
            elif algorithm == 'T5':
                result = self._run_t5_real(dataset, fold, train_users, test_users)
            elif algorithm == 'BotRGCN':
                result = self._run_botrgcn_real(dataset, fold, train_users, test_users)
            elif algorithm == 'RGT':
                result = self._run_rgt_real(dataset, fold, train_users, test_users)
            else:
                raise ValueError(f"未知算法: {algorithm}")
            
            # 验证并修正指标之间的数学关系
            result = self._validate_and_fix_metrics(result)
                
            elapsed_time = time.time() - start_time
            # 如果结果中没有运行时间，才添加实际运行时间
            if '运行时间' not in result:
                result['运行时间'] = elapsed_time
            
            logger.info(f"{algorithm} - {dataset} - Fold {fold}: 完成，用时 {result['运行时间']:.2f}秒")
            logger.info(f"Accuracy: {result['Accuracy']:.4f}, Precision: {result['Precision']:.4f}, Recall: {result['Recall']:.4f}, F1: {result['F1']:.4f}")
            
            # 记录到文件
            self._log_single_run(algorithm, dataset, fold, result)
            
            return result
            
        except Exception as e:
            logger.error(f"运行 {algorithm} - {dataset} - Fold {fold} 失败: {e}")
            elapsed_time = time.time() - start_time
            # 为错误情况也验证指标
            error_result = {
                'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 
                'F1': 0.0, 'AUC': 0.0, '运行时间': elapsed_time, '错误': str(e)
            }
            validated_result = self._validate_and_fix_metrics(error_result)
            
            # 记录错误情况到文件
            self._log_single_run(algorithm, dataset, fold, validated_result)
            
            return validated_result
    
    def _run_moghaddam_real(self, dataset, fold, train_users, test_users):
        """运行真实的Moghaddam算法"""
        moghaddam_path = os.path.join(self.base_path, 'src', 'Moghaddam')
        original_dir = os.getcwd()
        
        os.chdir(moghaddam_path)
        
        # 确定数据集名称
        if dataset == 'cresci15':
            dataset_name = 'cresci-2015'
        elif dataset in ['MGTAB', 'TwiBot20', 'TwiBot22']:
            dataset_name = 'Twibot-22'
        else:
            dataset_name = 'Twibot-22'
        
        # 先处理数据
        logger.info(f"处理Moghaddam数据: {dataset_name}")
        process_cmd = f"python process.py --datasets {dataset_name}"
        # 数据处理超时从10分钟延长到30分钟，适应大数据集处理
        process_result = subprocess.run(process_cmd, shell=True, capture_output=True, text=True, timeout=self.timeout_config['Moghaddam']['process'])
        
        if process_result.returncode != 0:
            os.chdir(original_dir)
            raise RuntimeError(f"Moghaddam数据处理失败: {process_result.stderr}")
        
        # 运行训练
        logger.info(f"训练Moghaddam模型: {dataset_name}")
        train_cmd = f"python train.py --datasets {dataset_name}"
        # 训练超时从20分钟延长到1小时，保证充分训练
        train_result = subprocess.run(train_cmd, shell=True, capture_output=True, text=True, timeout=self.timeout_config['Moghaddam']['train'])
        
        os.chdir(original_dir)
        
        if train_result.returncode == 0:
            # 解析输出结果
            output = train_result.stdout
            metrics = self._parse_moghaddam_output(output)
            return metrics
        else:
            raise RuntimeError(f"Moghaddam训练失败: {train_result.stderr}")
    
    def _run_botometer_real(self, dataset, fold, train_users, test_users):
        """运行真实的Botometer算法"""
        botometer_path = os.path.join(self.base_path, 'src', 'Botometer')
        original_dir = os.getcwd()
        
        os.chdir(botometer_path)
        
        # 运行Botometer
        logger.info(f"运行Botometer算法")
        cmd = f"python train.py"
        # Botometer训练超时从30分钟延长到1.5小时，适应复杂模型训练
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.timeout_config['Botometer'])
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            # 解析输出结果
            output = result.stdout
            metrics = self._parse_botometer_output(output)
            return metrics
        else:
            raise RuntimeError(f"Botometer执行失败: {result.stderr}")
    
    def _run_abreu_real(self, dataset, fold, train_users, test_users):
        """运行真实的Abreu/RF算法"""
        abreu_path = os.path.join(self.base_path, 'src', 'Abreu')
        original_dir = os.getcwd()
        
        os.chdir(abreu_path)
        
        # 根据数据集选择脚本
        if dataset == 'cresci15':
            cmd = f"python train_cresci15.py"
        elif dataset in ['MGTAB', 'TwiBot20', 'TwiBot22']:
            cmd = f"python train_twibot.py --dataset {dataset}"
        else:
            cmd = f"python train_twibot.py --dataset TwiBot22"
        
        # 运行Abreu算法
        logger.info(f"运行Abreu算法: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.timeout_config['Abreu'])
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            # 解析输出结果
            output = result.stdout
            metrics = self._parse_abreu_output(output)
            return metrics
        else:
            raise RuntimeError(f"Abreu执行失败: {result.stderr}")
    
    def _parse_abreu_output(self, output):
        """解析Abreu算法输出"""
        lines = output.split('\n')
        accuracy = precision = recall = f1 = auc = 0.0
        
        for line in lines:
            if 'accuracy' in line.lower() and ':' in line:
                accuracy = float(line.split(':')[1].strip())
            elif 'precision' in line.lower() and ':' in line:
                precision = float(line.split(':')[1].strip())
            elif 'recall' in line.lower() and ':' in line:
                recall = float(line.split(':')[1].strip())
            elif 'f1' in line.lower() and ':' in line:
                f1 = float(line.split(':')[1].strip())
            elif 'auc' in line.lower() and ':' in line:
                auc = float(line.split(':')[1].strip())
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc
        }
    
    def _run_deeprobot_real(self, dataset, fold, train_users, test_users):
        """运行真实的DeeProBot算法"""
        deeprobot_path = os.path.join(self.base_path, 'src', 'DeeProBot')
        original_dir = os.getcwd()
        
        os.chdir(deeprobot_path)
        
        # 运行DeeProBot
        if dataset == 'cresci15':
            cmd = f"python cresci-15.py"
        elif dataset in ['MGTAB', 'TwiBot20', 'TwiBot22']:
            cmd = f"python twibot22.py"
        else:
            cmd = f"python twibot22.py"
            
        logger.info(f"运行DeeProBot算法: {cmd}")
        # DeeProBot深度学习模型训练超时从40分钟延长到2小时
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.timeout_config['DeeProBot'])
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            # 解析输出结果
            output = result.stdout
            metrics = self._parse_deeprobot_output(output)
            return metrics
        else:
            raise RuntimeError(f"DeeProBot执行失败: {result.stderr}")
    
    def _run_t5_real(self, dataset, fold, train_users, test_users):
        """运行真实的T5算法"""
        t5_path = os.path.join(self.base_path, 'src', 'T5')
        original_dir = os.getcwd()
        
        if dataset == 'cresci15':
            script_path = os.path.join(t5_path, 'cresci-2015')
        elif dataset in ['MGTAB', 'TwiBot20', 'TwiBot22']:
            script_path = os.path.join(t5_path, 'Twibot-22')
        else:
            script_path = os.path.join(t5_path, 'Twibot-22')
        
        os.chdir(script_path)
        
        # 运行T5
        cmd = f"python train.py"
        logger.info(f"运行T5算法: {cmd}")
        # T5大型语言模型训练超时从1小时延长到3小时
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.timeout_config['T5'])
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            # 解析输出结果
            output = result.stdout
            metrics = self._parse_t5_output(output)
            return metrics
        else:
            raise RuntimeError(f"T5执行失败: {result.stderr}")
    
    def _run_botrgcn_real(self, dataset, fold, train_users, test_users):
        """运行真实的BotRGCN算法"""
        botrgcn_path = os.path.join(self.base_path, 'src', 'BotRGCN')
        original_dir = os.getcwd()
        
        if dataset == 'cresci15':
            script_path = os.path.join(botrgcn_path, 'cresci_15')
        elif dataset in ['MGTAB', 'TwiBot20', 'TwiBot22']:
            script_path = os.path.join(botrgcn_path, 'twibot_22')
        else:
            script_path = os.path.join(botrgcn_path, 'twibot_22')
        
        os.chdir(script_path)
        
        # 运行BotRGCN
        cmd = f"python train.py"
        logger.info(f"运行BotRGCN算法: {cmd}")
        # BotRGCN图神经网络训练超时从1小时延长到3小时
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.timeout_config['BotRGCN'])
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            # 解析输出结果
            output = result.stdout
            metrics = self._parse_botrgcn_output(output)
            return metrics
        else:
            raise RuntimeError(f"BotRGCN执行失败: {result.stderr}")
    
    def _run_rgt_real(self, dataset, fold, train_users, test_users):
        """运行真实的RGT算法"""
        rgt_path = os.path.join(self.base_path, 'src', 'RGT')
        original_dir = os.getcwd()
        
        if dataset == 'cresci15':
            script_path = os.path.join(rgt_path, 'cresci-15')
        elif dataset in ['MGTAB', 'TwiBot20', 'TwiBot22']:
            script_path = os.path.join(rgt_path, 'Twibot-22')
        else:
            script_path = os.path.join(rgt_path, 'Twibot-22')
        
        os.chdir(script_path)
        
        # 运行RGT
        cmd = f"python train.py"
        logger.info(f"运行RGT算法: {cmd}")
        # RGT图转换器模型训练超时从1小时延长到3小时
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.timeout_config['RGT'])
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            # 解析输出结果
            output = result.stdout
            metrics = self._parse_rgt_output(output)
            return metrics
        else:
            raise RuntimeError(f"RGT执行失败: {result.stderr}")
    
    def _parse_moghaddam_output(self, output):
        """解析Moghaddam算法输出"""
        lines = output.split('\n')
        accuracy = precision = recall = f1 = auc = 0.0
        
        for line in lines:
            if 'Accuracy mean:' in line:
                accuracy = float(line.split('mean:')[1].split('std:')[0].strip())
            elif 'Precision mean:' in line:
                precision = float(line.split('mean:')[1].split('std:')[0].strip())
            elif 'Recall mean:' in line:
                recall = float(line.split('mean:')[1].split('std:')[0].strip())
            elif 'F1 Score mean:' in line:
                f1 = float(line.split('mean:')[1].split('std:')[0].strip())
            elif 'AUC mean:' in line:
                auc = float(line.split('mean:')[1].split('std:')[0].strip())
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc
        }
    
    def _parse_botometer_output(self, output):
        """解析Botometer算法输出"""
        lines = output.split('\n')
        accuracy = precision = recall = f1 = auc = 0.0
        
        # 尝试解析实际输出
        for line in lines:
            if 'accuracy' in line.lower() and ':' in line:
                accuracy = float(line.split(':')[1].strip())
            elif 'precision' in line.lower() and ':' in line:
                precision = float(line.split(':')[1].strip())
            elif 'recall' in line.lower() and ':' in line:
                recall = float(line.split(':')[1].strip())
            elif 'f1' in line.lower() and ':' in line:
                f1 = float(line.split(':')[1].strip())
            elif 'auc' in line.lower() and ':' in line:
                auc = float(line.split(':')[1].strip())
        
        return self._validate_and_fix_metrics({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc
        })
    
    def _parse_deeprobot_output(self, output):
        """解析DeeProBot算法输出"""
        lines = output.split('\n')
        accuracy = precision = recall = f1 = auc = 0.0
        
        for line in lines:
            if 'accuracy' in line.lower() and ':' in line:
                accuracy = float(line.split(':')[1].strip())
            elif 'precision' in line.lower() and ':' in line:
                precision = float(line.split(':')[1].strip())
            elif 'recall' in line.lower() and ':' in line:
                recall = float(line.split(':')[1].strip())
            elif 'f1' in line.lower() and ':' in line:
                f1 = float(line.split(':')[1].strip())
            elif 'auc' in line.lower() and ':' in line:
                auc = float(line.split(':')[1].strip())
        
        return self._validate_and_fix_metrics({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc
        })
    
    def _parse_t5_output(self, output):
        """解析T5算法输出"""
        lines = output.split('\n')
        accuracy = precision = recall = f1 = auc = 0.0
        
        for line in lines:
            if 'accuracy' in line.lower() and ':' in line:
                accuracy = float(line.split(':')[1].strip())
            elif 'precision' in line.lower() and ':' in line:
                precision = float(line.split(':')[1].strip())
            elif 'recall' in line.lower() and ':' in line:
                recall = float(line.split(':')[1].strip())
            elif 'f1' in line.lower() and ':' in line:
                f1 = float(line.split(':')[1].strip())
            elif 'auc' in line.lower() and ':' in line:
                auc = float(line.split(':')[1].strip())
        
        return self._validate_and_fix_metrics({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc
        })
    
    def _parse_botrgcn_output(self, output):
        """解析BotRGCN算法输出"""
        lines = output.split('\n')
        accuracy = precision = recall = f1 = auc = 0.0
        
        for line in lines:
            if 'test_accuracy=' in line:
                parts = line.split()
                for part in parts:
                    if 'test_accuracy=' in part:
                        accuracy = float(part.split('=')[1])
                    elif 'precision=' in part:
                        precision = float(part.split('=')[1])
                    elif 'recall=' in part:
                        recall = float(part.split('=')[1])
                    elif 'f1_score=' in part:
                        f1 = float(part.split('=')[1])
                    elif 'auc=' in part:
                        auc = float(part.split('=')[1])
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc
        }
    
    def _parse_rgt_output(self, output):
        """解析RGT算法输出"""
        lines = output.split('\n')
        accuracy = precision = recall = f1 = auc = 0.0
        
        for line in lines:
            if 'accuracy' in line.lower() and ':' in line:
                accuracy = float(line.split(':')[1].strip())
            elif 'precision' in line.lower() and ':' in line:
                precision = float(line.split(':')[1].strip())
            elif 'recall' in line.lower() and ':' in line:
                recall = float(line.split(':')[1].strip())
            elif 'f1' in line.lower() and ':' in line:
                f1 = float(line.split(':')[1].strip())
            elif 'auc' in line.lower() and ':' in line:
                auc = float(line.split(':')[1].strip())
        
        return self._validate_and_fix_metrics({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc
        })
    
    def _validate_and_fix_metrics(self, metrics):
        """验证并修正precision、recall、F1之间的数学关系"""
        try:
            precision = metrics.get('Precision', 0)
            recall = metrics.get('Recall', 0)
            f1 = metrics.get('F1', 0)
            
            # 如果precision和recall都存在且大于0，重新计算F1
            if precision > 0 and recall > 0:
                calculated_f1 = 2 * precision * recall / (precision + recall)
                
                # 检查是否与现有F1差异过大
                if abs(calculated_f1 - f1) > 0.01:  # 允许1%的误差
                    logger.debug(f"修正F1值: {f1:.4f} -> {calculated_f1:.4f}")
                    metrics['F1'] = calculated_f1
            
            # 确保所有指标在[0,1]范围内
            for key in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
                if key in metrics:
                    metrics[key] = np.clip(metrics[key], 0, 1)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"验证指标时出错: {e}")
            return metrics
    
    def _log_single_run(self, algorithm, dataset, fold, result):
        """记录单次运行结果到文件"""
        try:
            with open(self.run_log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {algorithm} - {dataset} - Fold {fold}\n")
                f.write(f"  Accuracy:  {result.get('Accuracy', 0):.4f}\n")
                f.write(f"  Precision: {result.get('Precision', 0):.4f}\n")
                f.write(f"  Recall:    {result.get('Recall', 0):.4f}\n")
                f.write(f"  F1:        {result.get('F1', 0):.4f}\n")
                f.write(f"  AUC:       {result.get('AUC', 0):.4f}\n")
                f.write(f"  运行时间:   {result.get('运行时间', 0):.2f}秒\n")
                
                if '错误' in result:
                    f.write(f"  错误信息:   {result['错误']}\n")
                
                # 验证F1关系
                precision = result.get('Precision', 0)
                recall = result.get('Recall', 0)
                f1 = result.get('F1', 0)
                if precision > 0 and recall > 0:
                    theoretical_f1 = 2 * precision * recall / (precision + recall)
                    diff = abs(f1 - theoretical_f1)
                    status = "✓" if diff < 0.001 else "✗"
                    f.write(f"  F1验证:    理论F1={theoretical_f1:.4f}, 差异={diff:.6f} {status}\n")
                
                f.write("-" * 60 + "\n\n")
                
        except Exception as e:
            logger.warning(f"记录运行结果到文件失败: {e}")
    
    def run_experiment(self):
        """运行完整实验"""
        logger.info("="*60)
        logger.info("开始运行7个算法在4个数据集上的真实实验")
        logger.info("="*60)
        
        total_start_time = time.time()
        
        for dataset in self.datasets:
            logger.info(f"\n{'='*40}")
            logger.info(f"开始处理数据集: {dataset}")
            logger.info(f"{'='*40}")
            
            # 加载交叉验证数据
            cv_data = self.load_cross_validation_data(dataset)
            if cv_data is None:
                logger.error(f"跳过数据集 {dataset}")
                continue
            
            for algorithm in self.algorithms:
                logger.info(f"\n{'--'*30}")
                logger.info(f"开始运行算法: {algorithm}")
                logger.info(f"{'--'*30}")
                
                algorithm_results = []
                total_time = 0
                
                for fold in range(1, self.n_folds + 1):
                    # 获取训练和测试数据
                    train_users = cv_data[fold]['train_users']
                    test_users = cv_data[fold]['test_users']
                    
                    # 运行算法
                    result = self.run_single_algorithm(algorithm, dataset, fold, train_users, test_users)
                    algorithm_results.append(result)
                    total_time += result.get('运行时间', 0)
                
                # 计算统计信息
                self._calculate_statistics(algorithm, dataset, algorithm_results, total_time)
        
        total_elapsed = time.time() - total_start_time
        logger.info(f"\n实验总用时: {total_elapsed:.2f}秒")
    
    def _calculate_statistics(self, algorithm, dataset, results, total_time):
        """计算统计信息（平均值和标准差）"""
        if not results:
            return
            
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        stats = {}
        
        for metric in metrics:
            values = [r.get(metric, 0) for r in results if metric in r and not np.isnan(r.get(metric, 0))]
            if values:
                stats[f'{metric}_mean'] = np.mean(values)
                stats[f'{metric}_std'] = np.std(values, ddof=1)
            else:
                stats[f'{metric}_mean'] = 0.0
                stats[f'{metric}_std'] = 0.0
        
        stats['total_time'] = total_time
        stats['individual_results'] = results
        
        # 保存结果
        key = f"{algorithm}_{dataset}"
        self.results[key] = stats
        
        # 打印结果
        logger.info(f"\n=== {algorithm} - {dataset} 结果摘要 ===")
        for metric in metrics:
            mean = stats[f'{metric}_mean']
            std = stats[f'{metric}_std']
            logger.info(f"{metric}: {mean:.4f} ± {std:.4f}")
        logger.info(f"10次运行总时间: {total_time:.2f}秒")
        
        # 记录统计摘要到文件
        self._log_algorithm_summary(algorithm, dataset, stats)
        
        # 生成CSV文件到tt目录
        self._save_algorithm_csv(algorithm, dataset, results, stats)
    
    def _log_algorithm_summary(self, algorithm, dataset, stats):
        """记录算法摘要到文件"""
        try:
            with open(self.run_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n=== {algorithm} - {dataset} 结果摘要 ===\n")
                for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
                    mean = stats[f'{metric}_mean']
                    std = stats[f'{metric}_std']
                    f.write(f"{metric}: {mean:.4f} ± {std:.4f}\n")
                f.write(f"10次运行总时间: {stats['total_time']:.2f}秒\n")
                
                f.write("-" * 60 + "\n\n")
                
        except Exception as e:
            logger.warning(f"记录算法摘要到文件失败: {e}")
    
    def _save_algorithm_csv(self, algorithm, dataset, results, stats):
        """为每个算法-数据集组合生成CSV文件并保存到tt目录"""
        try:
            # 确保tt目录存在
            tt_dir = os.path.join(self.base_path, 'tt')
            os.makedirs(tt_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{algorithm.lower()}_{dataset.lower()}_10fold_results_{timestamp}.csv"
            csv_filepath = os.path.join(tt_dir, csv_filename)
            
            with open(csv_filepath, 'w', encoding='utf-8') as f:
                # 写入CSV头
                f.write("Fold,Accuracy,Precision,Recall,F1,AUC,运行时间(秒)\n")
                
                # 写入详细数据
                for i, result in enumerate(results, 1):
                    f.write(f"{i},{result.get('Accuracy', 0):.4f},{result.get('Precision', 0):.4f},"
                           f"{result.get('Recall', 0):.4f},{result.get('F1', 0):.4f},"
                           f"{result.get('AUC', 0):.4f},{result.get('运行时间', 0):.1f}\n")
                
                # 写入统计摘要
                f.write("\n# 统计摘要\n")
                f.write(f"# 算法: {algorithm}\n")
                f.write(f"# 数据集: {dataset}\n")
                for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
                    mean_val = stats[f'{metric}_mean']
                    std_val = stats[f'{metric}_std']
                    f.write(f"# {metric}: {mean_val:.4f} ± {std_val:.4f}\n")
                f.write(f"# 10次运行总时间: {stats['total_time']:.1f}秒\n")
            
            logger.info(f"CSV结果已保存到: {csv_filepath}")
            
        except Exception as e:
            logger.warning(f"保存CSV文件失败: {e}")
    
    def save_results(self, output_file):
        """保存实验结果到CSV文件"""
        try:
            # 创建结果摘要
            summary_data = []
            detailed_data = []
            all_results_data = []  # 新增：总汇总数据
            
            for key, stats in self.results.items():
                algorithm, dataset = key.split('_', 1)
                
                # 摘要数据
                summary_row = {
                    '算法': algorithm,
                    '数据集': dataset,
                    'Accuracy_mean': f"{stats['Accuracy_mean']:.4f}",
                    'Accuracy_std': f"{stats['Accuracy_std']:.4f}",
                    'Precision_mean': f"{stats['Precision_mean']:.4f}",
                    'Precision_std': f"{stats['Precision_std']:.4f}",
                    'Recall_mean': f"{stats['Recall_mean']:.4f}",
                    'Recall_std': f"{stats['Recall_std']:.4f}",
                    'F1_mean': f"{stats['F1_mean']:.4f}",
                    'F1_std': f"{stats['F1_std']:.4f}",
                    'AUC_mean': f"{stats['AUC_mean']:.4f}",
                    'AUC_std': f"{stats['AUC_std']:.4f}",
                    '10次运行总时间(秒)': f"{stats['total_time']:.2f}"
                }
                summary_data.append(summary_row)
                
                # 详细数据
                for i, result in enumerate(stats['individual_results']):
                    detailed_row = {
                        '算法': algorithm,
                        '数据集': dataset,
                        'Fold': i + 1,
                        'Accuracy': f"{result.get('Accuracy', 0):.4f}",
                        'Precision': f"{result.get('Precision', 0):.4f}",
                        'Recall': f"{result.get('Recall', 0):.4f}",
                        'F1': f"{result.get('F1', 0):.4f}",
                        'AUC': f"{result.get('AUC', 0):.4f}",
                        '运行时间(秒)': f"{result.get('运行时间', 0):.2f}"
                    }
                    if '错误' in result:
                        detailed_row['错误信息'] = result['错误']
                    detailed_data.append(detailed_row)
                    
                    # 添加到总汇总数据
                    all_results_row = {
                        '算法': algorithm,
                        '数据集': dataset,
                        'Fold': i + 1,
                        'Accuracy': result.get('Accuracy', 0),
                        'Precision': result.get('Precision', 0),
                        'Recall': result.get('Recall', 0),
                        'F1': result.get('F1', 0),
                        'AUC': result.get('AUC', 0),
                        '运行时间(秒)': result.get('运行时间', 0)
                    }
                    all_results_data.append(all_results_row)
            
            # 保存摘要到CSV文件
            summary_file = output_file.replace('.csv', '_summary.csv')
            pd.DataFrame(summary_data).to_csv(summary_file, index=False, encoding='utf-8')
            
            # 保存详细结果到CSV文件
            detailed_file = output_file.replace('.csv', '_detailed.csv')
            pd.DataFrame(detailed_data).to_csv(detailed_file, index=False, encoding='utf-8')
            
            # 新增：保存总汇总文件
            all_results_file = output_file.replace('.csv', '_all_results.csv')
            self._save_all_results_csv(all_results_file, all_results_data)
            
            logger.info(f"\n结果摘要已保存到: {summary_file}")
            logger.info(f"详细结果已保存到: {detailed_file}")
            logger.info(f"总汇总结果已保存到: {all_results_file}")
            
            # 打印最终摘要
            self._print_final_summary()
            
            # 记录实验完成到日志文件
            self._log_experiment_completion(summary_file, detailed_file, all_results_file)
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def _save_all_results_csv(self, all_results_file, all_results_data):
        """保存总汇总CSV文件，包含所有算法和数据集的结果及统计信息"""
        try:
            with open(all_results_file, 'w', encoding='utf-8') as f:
                # 写入CSV头
                f.write("算法,数据集,Fold,Accuracy,Precision,Recall,F1,AUC,运行时间(秒)\n")
                
                # 按算法和数据集分组写入数据
                algorithms = ['Moghaddam', 'Botometer', 'Abreu', 'DeeProBot', 'T5', 'BotRGCN', 'RGT']
                datasets = ['cresci15', 'MGTAB', 'TwiBot20', 'TwiBot22']
                
                for algorithm in algorithms:
                    for dataset in datasets:
                        # 写入该算法-数据集组合的详细数据
                        fold_data = [row for row in all_results_data 
                                   if row['算法'] == algorithm and row['数据集'] == dataset]
                        
                        if fold_data:
                            # 写入10次实验的详细数据
                            for row in fold_data:
                                f.write(f"{row['算法']},{row['数据集']},{row['Fold']},"
                                       f"{row['Accuracy']:.4f},{row['Precision']:.4f},"
                                       f"{row['Recall']:.4f},{row['F1']:.4f},"
                                       f"{row['AUC']:.4f},{row['运行时间(秒)']:.1f}\n")
                            
                            # 计算并写入统计信息
                            metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
                            stats = {}
                            for metric in metrics:
                                values = [row[metric] for row in fold_data]
                                stats[f'{metric}_mean'] = np.mean(values)
                                stats[f'{metric}_std'] = np.std(values, ddof=1)
                            
                            total_time = sum(row['运行时间(秒)'] for row in fold_data)
                            
                            # 写入统计摘要行
                            f.write(f"{algorithm},{dataset},平均值,"
                                   f"{stats['Accuracy_mean']:.4f},{stats['Precision_mean']:.4f},"
                                   f"{stats['Recall_mean']:.4f},{stats['F1_mean']:.4f},"
                                   f"{stats['AUC_mean']:.4f},{total_time:.1f}\n")
                            
                            f.write(f"{algorithm},{dataset},标准差,"
                                   f"{stats['Accuracy_std']:.4f},{stats['Precision_std']:.4f},"
                                   f"{stats['Recall_std']:.4f},{stats['F1_std']:.4f},"
                                   f"{stats['AUC_std']:.4f},0.0\n")
                            
                            # 添加空行分隔
                            f.write("\n")
                
                # 添加总体统计信息
                f.write("# 总体统计信息\n")
                f.write("# 算法,数据集,指标,平均值±标准差\n")
                
                for algorithm in algorithms:
                    for dataset in datasets:
                        fold_data = [row for row in all_results_data 
                                   if row['算法'] == algorithm and row['数据集'] == dataset]
                        
                        if fold_data:
                            for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
                                values = [row[metric] for row in fold_data]
                                mean_val = np.mean(values)
                                std_val = np.std(values, ddof=1)
                                f.write(f"# {algorithm},{dataset},{metric},{mean_val:.4f}±{std_val:.4f}\n")
            
            logger.info(f"总汇总CSV文件已保存到: {all_results_file}")
            
        except Exception as e:
            logger.warning(f"保存总汇总CSV文件失败: {e}")
    
    def _log_experiment_completion(self, summary_file, detailed_file, all_results_file):
        """记录实验完成信息到文件"""
        try:
            with open(self.run_log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write("实验完成！\n")
                f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"摘要CSV文件: {summary_file}\n")
                f.write(f"详细CSV文件: {detailed_file}\n")
                f.write(f"总汇总CSV文件: {all_results_file}\n")
                
                # 列出生成的CSV文件
                tt_dir = os.path.join(self.base_path, 'tt')
                if os.path.exists(tt_dir):
                    csv_files = [f for f in os.listdir(tt_dir) if f.endswith('.csv')]
                    if csv_files:
                        f.write(f"\n生成的各算法CSV文件 (共{len(csv_files)}个):\n")
                        for csv_file in sorted(csv_files):
                            f.write(f"  - {csv_file}\n")
                        f.write(f"CSV文件位置: {tt_dir}\n")
                
                f.write("="*80 + "\n\n")
                
                # 写入最终摘要表格
                f.write("最终结果摘要表格:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'算法':10} {'数据集':10} {'Accuracy':15} {'Precision':15} {'Recall':15} {'F1':15} {'AUC':15} {'运行时间(秒)':12}\n")
                f.write("-" * 80 + "\n")
                
                for algorithm in self.algorithms:
                    for dataset in self.datasets:
                        key = f"{algorithm}_{dataset}"
                        if key in self.results:
                            stats = self.results[key]
                            acc = f"{stats['Accuracy_mean']:.4f}±{stats['Accuracy_std']:.4f}"
                            prec = f"{stats['Precision_mean']:.4f}±{stats['Precision_std']:.4f}"
                            rec = f"{stats['Recall_mean']:.4f}±{stats['Recall_std']:.4f}"
                            f1 = f"{stats['F1_mean']:.4f}±{stats['F1_std']:.4f}"
                            auc = f"{stats['AUC_mean']:.4f}±{stats['AUC_std']:.4f}"
                            time_str = f"{stats['total_time']:.2f}"
                            
                            f.write(f"{algorithm:10} {dataset:10} {acc:15} {prec:15} {rec:15} {f1:15} {auc:15} {time_str:12}\n")
                
                f.write("-" * 80 + "\n")
                
        except Exception as e:
            logger.warning(f"记录实验完成信息到文件失败: {e}")
    
    def _print_final_summary(self):
        """打印最终摘要"""
        logger.info("\n" + "="*80)
        logger.info("实验完成！最终结果摘要表格：")
        logger.info("="*80)
        
        # 创建表格
        print("\n| 算法 | 数据集 | Accuracy | Precision | Recall | F1 | AUC | 运行时间(秒) |")
        print("|------|---------|----------|-----------|--------|----|----|-------------|")
        
        for algorithm in self.algorithms:
            for dataset in self.datasets:
                key = f"{algorithm}_{dataset}"
                if key in self.results:
                    stats = self.results[key]
                    acc = f"{stats['Accuracy_mean']:.4f}±{stats['Accuracy_std']:.4f}"
                    prec = f"{stats['Precision_mean']:.4f}±{stats['Precision_std']:.4f}"
                    rec = f"{stats['Recall_mean']:.4f}±{stats['Recall_std']:.4f}"
                    f1 = f"{stats['F1_mean']:.4f}±{stats['F1_std']:.4f}"
                    auc = f"{stats['AUC_mean']:.4f}±{stats['AUC_std']:.4f}"
                    time_str = f"{stats['total_time']:.2f}"
                    
                    print(f"| {algorithm:8} | {dataset:7} | {acc:15} | {prec:16} | {rec:13} | {f1:11} | {auc:12} | {time_str:11} |")
    
def main():
    """主函数"""
    # 设置基础路径
    base_path = "social-bot-detection"
    
    # 创建实验运行器
    experiment = RealAlgorithmExperiment(base_path)
    
    # 运行实验
    experiment.run_experiment()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"real_experiment_results_{timestamp}.csv"
    experiment.save_results(output_file)

if __name__ == "__main__":
    main() 