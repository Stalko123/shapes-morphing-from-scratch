#!/usr/bin/env python3
"""
Script pour visualiser l'évolution des différents schedulers de learning rate
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def create_scheduler_plots():
    """Crée des graphiques pour chaque type de scheduler"""
    
    # Configuration générale
    n_epochs = 100
    initial_lr = 0.001
    epochs = np.arange(1, n_epochs + 1)
    
    # Créer le dossier pour sauvegarder les images
    output_dir = "scheduler_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Style des graphiques
    plt.style.use('default')
    fig_size = (12, 8)
    
    print("Génération des visualisations des schedulers...")
    
    # 1. StepLR Scheduler
    print("1. Générant StepLR scheduler...")
    step_size = 20
    gamma = 0.5
    lr_step = []
    current_lr = initial_lr
    
    for epoch in epochs:
        if (epoch - 1) % step_size == 0 and epoch > 1:
            current_lr *= gamma
        lr_step.append(current_lr)
    
    plt.figure(figsize=fig_size)
    plt.plot(epochs, lr_step, 'b-', linewidth=3, label='StepLR')
    plt.title(f'StepLR Scheduler\nDécroissance par paliers (step_size={step_size}, gamma={gamma})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.yscale('log')
    
    # Ajouter des annotations pour les paliers
    for i, step_epoch in enumerate(range(step_size, n_epochs, step_size)):
        if step_epoch < n_epochs:
            plt.axvline(x=step_epoch, color='red', linestyle='--', alpha=0.5)
            plt.annotate(f'LR × {gamma}', xy=(step_epoch, lr_step[step_epoch-1]), 
                        xytext=(step_epoch+5, lr_step[step_epoch-1]*2),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/step_lr_scheduler.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ExponentialLR Scheduler
    print("2. Générant ExponentialLR scheduler...")
    gamma_exp = 0.98
    lr_exp = [initial_lr * (gamma_exp ** (epoch - 1)) for epoch in epochs]
    
    plt.figure(figsize=fig_size)
    plt.plot(epochs, lr_exp, 'g-', linewidth=3, label='ExponentialLR')
    plt.title(f'ExponentialLR Scheduler\nDécroissance exponentielle (gamma={gamma_exp})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.yscale('log')
    
    # Annotation avec la formule
    plt.annotate(f'LR = {initial_lr} × {gamma_exp}^epoch', 
                xy=(50, lr_exp[49]), xytext=(60, lr_exp[30]),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=12, color='green',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/exponential_lr_scheduler.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. CosineAnnealingLR Scheduler
    print("3. Générant CosineAnnealingLR scheduler...")
    T_max = n_epochs
    eta_min = 1e-6
    lr_cosine = []
    
    for epoch in epochs:
        lr = eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * (epoch - 1) / T_max)) / 2
        lr_cosine.append(lr)
    
    plt.figure(figsize=fig_size)
    plt.plot(epochs, lr_cosine, 'r-', linewidth=3, label='CosineAnnealingLR')
    plt.title(f'CosineAnnealingLR Scheduler\nDécroissance en cosinus (T_max={T_max}, eta_min={eta_min})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.yscale('log')
    
    # Annotations
    plt.annotate('Début: LR max', xy=(1, initial_lr), xytext=(15, initial_lr/2),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=12, color='red')
    plt.annotate('Fin: LR min', xy=(n_epochs, eta_min), xytext=(n_epochs-20, eta_min*100),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cosine_annealing_lr_scheduler.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. CosineAnnealingWarmRestarts Scheduler
    print("4. Générant CosineAnnealingWarmRestarts scheduler...")
    T_0 = 10
    T_mult = 2
    eta_min_restart = 1e-6
    lr_cosine_restart = []
    
    epoch = 1
    current_cycle_length = T_0
    cycle_start = 1
    
    while epoch <= n_epochs:
        # Position dans le cycle actuel (0 à 1)
        cycle_progress = (epoch - cycle_start) / current_cycle_length
        
        # Calcul du LR avec cosine annealing
        lr = eta_min_restart + (initial_lr - eta_min_restart) * (1 + np.cos(np.pi * cycle_progress)) / 2
        lr_cosine_restart.append(lr)
        
        # Vérifier si on doit redémarrer
        if epoch - cycle_start + 1 >= current_cycle_length:
            cycle_start = epoch + 1
            current_cycle_length *= T_mult
        
        epoch += 1
    
    plt.figure(figsize=fig_size)
    plt.plot(epochs, lr_cosine_restart, 'm-', linewidth=3, label='CosineAnnealingWarmRestarts')
    plt.title(f'CosineAnnealingWarmRestarts Scheduler\nCosinus avec redémarrages (T_0={T_0}, T_mult={T_mult})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.yscale('log')
    
    # Marquer les redémarrages
    restart_epochs = [T_0]
    cycle_length = T_0
    while restart_epochs[-1] < n_epochs:
        cycle_length *= T_mult
        next_restart = restart_epochs[-1] + cycle_length
        if next_restart <= n_epochs:
            restart_epochs.append(next_restart)
    
    for i, restart_epoch in enumerate(restart_epochs):
        if restart_epoch <= n_epochs:
            plt.axvline(x=restart_epoch, color='orange', linestyle='--', alpha=0.7, linewidth=2)
            plt.annotate(f'Restart {i+1}', xy=(restart_epoch, initial_lr), 
                        xytext=(restart_epoch+3, initial_lr/3),
                        arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7),
                        fontsize=10, color='orange', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cosine_warm_restarts_scheduler.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Comparaison de tous les schedulers
    print("5. Générant comparaison de tous les schedulers...")
    plt.figure(figsize=(16, 10))
    
    plt.plot(epochs, [initial_lr] * len(epochs), 'k--', linewidth=2, alpha=0.5, label='Constant LR')
    plt.plot(epochs, lr_step, 'b-', linewidth=3, label=f'StepLR (step={step_size}, γ={gamma})')
    plt.plot(epochs, lr_exp, 'g-', linewidth=3, label=f'ExponentialLR (γ={gamma_exp})')
    plt.plot(epochs, lr_cosine, 'r-', linewidth=3, label=f'CosineAnnealingLR (T_max={T_max})')
    plt.plot(epochs, lr_cosine_restart, 'm-', linewidth=3, label=f'CosineWarmRestarts (T_0={T_0}, T_mult={T_mult})')
    
    plt.title('Comparaison des Schedulers de Learning Rate', fontsize=20, fontweight='bold')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Learning Rate', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, loc='upper right')
    plt.yscale('log')
    
    # Ajouter une zone de texte avec des explications
    explanation = (
        "Stratégies de scheduling du Learning Rate:\n\n"
        "• StepLR: Réduction par paliers fixes\n"
        "• ExponentialLR: Décroissance continue exponentielle\n"
        "• CosineAnnealing: Décroissance douce suivant un cosinus\n"
        "• CosineWarmRestarts: Cycles répétés avec redémarrages\n"
        "• Constant: Référence sans scheduling"
    )
    plt.text(0.02, 0.98, explanation, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scheduler_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Toutes les visualisations ont été sauvegardées dans le dossier '{output_dir}/'")
    print("\nFichiers générés:")
    print("1. step_lr_scheduler.png - StepLR avec paliers")
    print("2. exponential_lr_scheduler.png - ExponentialLR décroissance continue")
    print("3. cosine_annealing_lr_scheduler.png - CosineAnnealingLR décroissance douce")
    print("4. cosine_warm_restarts_scheduler.png - CosineWarmRestarts avec cycles")
    print("5. scheduler_comparison.png - Comparaison de tous les schedulers")
    
    return output_dir

if __name__ == "__main__":
    output_dir = create_scheduler_plots()
    print(f"\n📊 Regardez les graphiques dans le dossier: {output_dir}/")
