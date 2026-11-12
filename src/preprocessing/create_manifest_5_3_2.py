#!/usr/bin/env python3
"""
Tạo Manifest Dữ Liệu Train / Val / Test Theo Tỷ Lệ Chuẩn 5 / 3 / 2 (50% / 30% / 20%)
Kết hợp Dataset ASL Gốc và Dataset Tự Chụp Đa Thành Viên (Khai, Viet, Bao, Tien, Dat,...)
Đảm bảo mỗi dataset thành viên và từng nhãn (label) bên trong đều chia chuẩn 5/3/2.
"""

import os
import sys
import math
import shutil
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Tạo manifest phân chia 5/3/2 theo từng thành viên và từng label")
    parser.add_argument("--raw_csv", default="data/landmarks/landmarks_raw.csv",
                        help="Đường dẫn CSV landmarks ASL gốc")
    parser.add_argument("--cap_csv", default="data/landmarks/landmarks_captured.csv",
                        help="Đường dẫn CSV landmarks tự chụp")
    parser.add_argument("--output_dir", default="data/hybrid_manifest",
                        help="Thư mục lưu train.csv, val.csv, test.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def allocate_5_3_2_for_member(member_df: pd.DataFrame, seed: int = 42):
    """
    Phân chia chính xác tỷ lệ 5/3/2 (50% Train, 30% Val, 20% Test) theo từng label
    của một thành viên, đồng thời đảm bảo tổng số mẫu của thành viên đó cũng đạt đúng 50% / 30% / 20%.
    """
    train_parts, val_parts, test_parts = [], [], []
    label_groups = []

    for label, grp in sorted(member_df.groupby("label"), key=lambda x: x[0]):
        n = len(grp)
        shuffled = grp.sample(frac=1.0, random_state=seed)
        exact_tr = n * 0.50
        exact_va = n * 0.30
        exact_te = n * 0.20

        # Sử dụng Largest Remainder Method (Phương pháp phần dư lớn nhất) cho từng label (n >= 3)
        if n >= 3:
            floors = [int(math.floor(exact_tr)), int(math.floor(exact_va)), int(math.floor(exact_te))]
            rems = [exact_tr - floors[0], exact_va - floors[1], exact_te - floors[2]]
            leftover = n - sum(floors)
            # Ưu tiên phần dư lớn nhất; nếu bằng nhau ưu tiên Train -> Val -> Test
            order = sorted(range(3), key=lambda idx: (rems[idx], -idx), reverse=True)
            for idx in order[:leftover]:
                floors[idx] += 1
            n_tr, n_va, n_te = floors[0], floors[1], floors[2]
            if n >= 5 and n_te == 0:
                n_te = 1
                if n_tr > 1:
                    n_tr -= 1
        elif n == 2:
            n_tr, n_va, n_te = 1, 1, 0
        else:
            n_tr, n_va, n_te = 1, 0, 0

        label_groups.append({
            "label": label,
            "df": shuffled,
            "n": n,
            "exact_tr": exact_tr,
            "exact_va": exact_va,
            "exact_te": exact_te,
            "n_tr": n_tr,
            "n_va": n_va,
            "n_te": n_te,
        })

    # Cân bằng phần dư giữa các label có số ảnh lẻ để tổng toàn bộ dataset của thành viên đạt đúng 50% / 30% / 20%
    total_n = sum(item["n"] for item in label_groups)
    if total_n >= 10:
        t_floors = [
            int(math.floor(total_n * 0.50)),
            int(math.floor(total_n * 0.30)),
            int(math.floor(total_n * 0.20)),
        ]
        t_rems = [
            total_n * 0.50 - t_floors[0],
            total_n * 0.30 - t_floors[1],
            total_n * 0.20 - t_floors[2],
        ]
        for idx in sorted(range(3), key=lambda i: (t_rems[i], -i), reverse=True)[: total_n - sum(t_floors)]:
            t_floors[idx] += 1
        target_tr, target_va, target_te = t_floors[0], t_floors[1], t_floors[2]

        for _ in range(total_n):
            cur_tr = sum(item["n_tr"] for item in label_groups)
            cur_va = sum(item["n_va"] for item in label_groups)
            cur_te = sum(item["n_te"] for item in label_groups)
            if (cur_tr, cur_va, cur_te) == (target_tr, target_va, target_te):
                break

            changed = False
            # Nếu Train dư và Test thiếu
            if cur_tr > target_tr and cur_te < target_te:
                cands = [x for x in label_groups if x["n_tr"] > math.floor(x["exact_tr"]) and x["n_te"] < math.ceil(x["exact_te"])]
                if cands:
                    best = max(cands, key=lambda x: (x["n_tr"] - x["exact_tr"]) + (x["exact_te"] - x["n_te"]))
                    best["n_tr"] -= 1
                    best["n_te"] += 1
                    changed = True
            # Nếu Train dư và Val thiếu
            if not changed and cur_tr > target_tr and cur_va < target_va:
                cands = [x for x in label_groups if x["n_tr"] > math.floor(x["exact_tr"]) and x["n_va"] < math.ceil(x["exact_va"])]
                if cands:
                    best = max(cands, key=lambda x: (x["n_tr"] - x["exact_tr"]) + (x["exact_va"] - x["n_va"]))
                    best["n_tr"] -= 1
                    best["n_va"] += 1
                    changed = True
            # Nếu Val dư và Test thiếu
            if not changed and cur_va > target_va and cur_te < target_te:
                cands = [x for x in label_groups if x["n_va"] > math.floor(x["exact_va"]) and x["n_te"] < math.ceil(x["exact_te"])]
                if cands:
                    best = max(cands, key=lambda x: (x["n_va"] - x["exact_va"]) + (x["exact_te"] - x["n_te"]))
                    best["n_va"] -= 1
                    best["n_te"] += 1
                    changed = True
            # Nếu Test dư và Train thiếu
            if not changed and cur_te > target_te and cur_tr < target_tr:
                cands = [x for x in label_groups if x["n_te"] > max(1, math.floor(x["exact_te"])) and x["n_tr"] < math.ceil(x["exact_tr"])]
                if cands:
                    best = max(cands, key=lambda x: (x["n_te"] - x["exact_te"]) + (x["exact_tr"] - x["n_tr"]))
                    best["n_te"] -= 1
                    best["n_tr"] += 1
                    changed = True
            if not changed:
                break

    for item in label_groups:
        shuffled = item["df"]
        n_tr = item["n_tr"]
        n_va = item["n_va"]
        train_parts.append(shuffled.iloc[:n_tr])
        val_parts.append(shuffled.iloc[n_tr:n_tr + n_va])
        test_parts.append(shuffled.iloc[n_tr + n_va:])

    return (
        pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(),
        pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(),
        pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(),
    )

def split_captured_dataframe(df_cap: pd.DataFrame, seed: int = 42):
    """
    Phân chia dataset tự chụp theo tỷ lệ 5/3/2 cho từng thành viên (Khai, Viet, Bao, Tien, Dat,...)
    và từng label bên trong mỗi thành viên.
    """
    if df_cap is None or len(df_cap) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if "member" not in df_cap.columns:
        df_cap = df_cap.copy()
        df_cap["member"] = "Khai"

    train_list, val_list, test_list = [], [], []
    for member, member_df in sorted(df_cap.groupby("member"), key=lambda x: x[0]):
        m_tr, m_va, m_te = allocate_5_3_2_for_member(member_df, seed=seed)
        train_list.append(m_tr)
        val_list.append(m_va)
        test_list.append(m_te)

    train_cap = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val_cap = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test_cap = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
    return train_cap, val_cap, test_cap

def create_manifest_5_3_2(raw_csv: str, cap_csv: str, output_dir: str, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sao lưu manifest cũ nếu chưa sao lưu
    for f in ["train.csv", "val.csv", "test.csv"]:
        old_path = os.path.join(output_dir, f)
        backup_path = os.path.join(output_dir, f.replace(".csv", "_backup_70_15_15.csv"))
        if os.path.exists(old_path) and not os.path.exists(backup_path):
            shutil.copy2(old_path, backup_path)
            print(f"Đã sao lưu manifest cũ: {backup_path}")

    # 2. Phân chia tập ASL gốc (50% / 30% / 20%) theo label
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Không tìm thấy file landmark gốc: {raw_csv}")
    
    df_raw = pd.read_csv(raw_csv)
    print(f"\n1. Đang nạp dataset ASL gốc: {len(df_raw):,} mẫu")
    
    # 50% Train, 50% Remaining
    train_raw, rem_raw = train_test_split(df_raw, test_size=0.50, stratify=df_raw['label'], random_state=seed)
    # 30% Val, 20% Test (tức 60% Val, 40% Test trong 50% còn lại)
    val_raw, test_raw = train_test_split(rem_raw, test_size=0.40, stratify=rem_raw['label'], random_state=seed)

    # 3. Phân chia tập ASL tự chụp (50% / 30% / 20%) theo từng thành viên & từng label
    df_cap = None
    if os.path.exists(cap_csv):
        df_cap = pd.read_csv(cap_csv)
        print(f"2. Đang nạp dataset tự chụp (custom_captured): {len(df_cap):,} mẫu")
        train_cap, val_cap, test_cap = split_captured_dataframe(df_cap, seed=seed)
    else:
        print(f"⚠️ Không tìm thấy file landmark tự chụp ({cap_csv}). Chỉ phân chia dataset gốc.")
        train_cap, val_cap, test_cap = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 4. Hợp nhất và XÁO TRỘN NGẪU NHIÊN (shuffle) tập dữ liệu gốc và tự chụp
    train_combined = pd.concat([train_raw, train_cap], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_combined   = pd.concat([val_raw, val_cap], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_combined  = pd.concat([test_raw, test_cap], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    train_combined.to_csv(train_path, index=False)
    val_combined.to_csv(val_path, index=False)
    test_combined.to_csv(test_path, index=False)

    total_combined = len(train_combined) + len(val_combined) + len(test_combined)

    print("\n" + "=" * 72)
    print("📊 KẾT QUẢ PHÂN CHIA TẬP DỮ LIỆU HYBRID MANIFEST (TỶ LỆ 5 / 3 / 2):")
    print("=" * 72)
    print(f"{'Tập':<12} | {'Tổng mẫu':<10} | {'Tỷ lệ':<8} | {'Gốc (Raw)':<12} | {'Tự chụp':<10}")
    print("-" * 72)
    print(f"{'Train (50%)':<12} | {len(train_combined):<10,} | {len(train_combined)/total_combined*100:6.2f}% | {len(train_raw):<12,} | {len(train_cap):<10,}")
    print(f"{'Val   (30%)':<12} | {len(val_combined):<10,} | {len(val_combined)/total_combined*100:6.2f}% | {len(val_raw):<12,} | {len(val_cap):<10,}")
    print(f"{'Test  (20%)':<12} | {len(test_combined):<10,} | {len(test_combined)/total_combined*100:6.2f}% | {len(test_raw):<12,} | {len(test_cap):<10,}")
    print("-" * 72)
    print(f"{'TỔNG CỘNG':<12} | {total_combined:<10,} | 100.00% | {len(df_raw):<12,} | {len(df_cap) if df_cap is not None else 0:<10,}")
    print("=" * 72)
    if df_cap is not None and 'member' in df_cap.columns:
        members = sorted(df_cap['member'].unique())
        print(f"👥 Chi tiết chia 5/3/2 theo từng dataset thành viên ({', '.join(members)}):")
        for m in members:
            m_tr = len(train_cap[train_cap['member'] == m]) if len(train_cap) > 0 else 0
            m_va = len(val_cap[val_cap['member'] == m]) if len(val_cap) > 0 else 0
            m_te = len(test_cap[test_cap['member'] == m]) if len(test_cap) > 0 else 0
            m_tot = m_tr + m_va + m_te
            n_labels = df_cap[df_cap['member'] == m]['label'].nunique()
            print(
                f"   * {m + '_dataset':<14} ({m_tot:3d} mẫu, {n_labels:2d} labels): "
                f"Train={m_tr:3d} ({m_tr/m_tot*100:5.1f}%) | "
                f"Val={m_va:3d} ({m_va/m_tot*100:5.1f}%) | "
                f"Test={m_te:3d} ({m_te/m_tot*100:5.1f}%)"
            )
    print(f"💾 File manifests đã ghi tại: {output_dir}")
    print("=" * 72 + "\n")

def main():
    args = parse_args()
    create_manifest_5_3_2(args.raw_csv, args.cap_csv, args.output_dir, seed=args.seed)

if __name__ == "__main__":
    main()
