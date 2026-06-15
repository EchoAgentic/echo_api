#SingleInstance Force

; Alt + Q déclenche l'action
!q::
    ; La commande brute pour PowerShell
    path_command := "cd 'C:\Users\nicog\OneDrive\Documents\identity'; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; clear"
    
    ; On met la commande dans ton presse-papier (Clipboard)
    Clipboard := path_command
    
    ; On attend que le presse-papier soit prêt
    ClipWait, 2
    
    ; On simule le "Coller" (Ctrl+V) et "Entrée"
    Send ^v
    Sleep, 100
    Send {Enter}
return